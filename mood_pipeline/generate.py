# 사용자 텍스트 + 방 크기(WxDxH) → SDXL+ControlNet(canny+depth)+IP-Adapter로
# 기존 mood_library 레퍼런스를 스타일/구조 가이드 삼아 후보 이미지 N장 생성
#
# Colab GPU 전제. 로컬(CPU-only)에서는 이 모듈을 import는 할 수 있지만
# _load_pipeline() 이후 실제 생성 호출은 diffusers/torch GPU 환경이 필요함.
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "mood_pipeline"

import gc
import hashlib
import json

import numpy as np

from .config import (
    BASE_NEGATIVE_PROMPT,
    CONTROLNET_CANNY_MODEL_ID,
    CONTROLNET_DEPTH_MODEL_ID,
    DEFAULT_GEN_PRESET,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_NUM_CANDIDATES,
    DEPTH_DETECTOR_MODEL_ID,
    GENERATION_CACHE_DIR,
    GENERATION_IMAGE_SIZE,
    GENERATION_OUTPUT_DIR,
    HIGH_CEILING_HEIGHT_M,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_SCALE,
    IP_ADAPTER_SUBFOLDER,
    IP_ADAPTER_WEIGHT_NAME,
    LOW_CEILING_HEIGHT_M,
    MOOD_GEN_PRESETS,
    MOOD_LIBRARY_DIR,
    ROOM_SIZE_BRACKETS,
    ROOM_SIZE_FALLBACK,
    SDXL_BASE_MODEL_ID,
    SDXL_VAE_ID,
)
from .search import (
    prepare_prompt_for_search,
    search_images_within_moods,
    search_mood_by_prompt,
)

# 1차 무드 매칭 후 2차 레퍼런스 풀을 합칠 후보 무드 개수 (1이면 top-1 무드로만 좁힘)
MOOD_POOL_TOP_K = 2

# ============================================================
# 1. 방 크기 스케일 가드 — "5x7 원룸에 대저택 거실" 같은 비현실적 결과 방지
# ============================================================


def room_scale_guard(width_m: float, depth_m: float, height_m: float) -> dict:
    # 바닥면적(㎡) 구간으로 방 스케일 문구 결정. 정밀한 치수 반영은 model2(3D 배치)가 담당하고
    # 여기서는 "이 방 크기대에 안 맞는 규모의 방"이 생성되지 않게 가드하는 역할만 한다.
    floor_area = width_m * depth_m

    prompt_suffix, negative_suffix, size_class = None, None, None
    for max_area, cls, suffix, neg in ROOM_SIZE_BRACKETS:
        if floor_area <= max_area:
            size_class, prompt_suffix, negative_suffix = cls, suffix, neg
            break
    if size_class is None:
        size_class = "large_open"
        prompt_suffix, negative_suffix = ROOM_SIZE_FALLBACK

    ceiling_note = ""
    if height_m < LOW_CEILING_HEIGHT_M:
        ceiling_note = ", low flat ceiling"
    elif height_m > HIGH_CEILING_HEIGHT_M:
        ceiling_note = ", tall ceiling"

    return {
        "floor_area_m2": round(floor_area, 2),
        "width_m": width_m,
        "depth_m": depth_m,
        "height_m": height_m,
        "size_class": size_class,
        "prompt_suffix": prompt_suffix + ceiling_note,
        "negative_suffix": negative_suffix,
    }


# ============================================================
# 2. 가이드(canny + depth) 추출 — Drive 캐싱
# ============================================================

_detector_cache: dict = {}


def _get_detectors():
    # controlnet_aux 전처리기는 무겁기 때문에 프로세스당 한 번만 로드
    if "canny" not in _detector_cache:
        from controlnet_aux import CannyDetector, MidasDetector

        _detector_cache["canny"] = CannyDetector()
        _detector_cache["depth"] = MidasDetector.from_pretrained(DEPTH_DETECTOR_MODEL_ID)
    return _detector_cache["canny"], _detector_cache["depth"]


def _free_detectors():
    # canny/depth 전처리기 + SDXL 파이프라인(IP-Adapter 포함)을 동시에 CPU에 올려두면
    # Colab 무료 T4(시스템 RAM ~12.7GB)에서 OOM으로 세션이 죽는다.
    # 가이드 추출이 끝나면 전처리기를 메모리에서 내려서 피크 사용량을 줄인다.
    _detector_cache.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class GuideExtractor:
    """레퍼런스 이미지 1장 → (canny, depth) 가이드 이미지. 결과는 cache_dir에 캐싱."""

    def __init__(self, cache_dir: Path | None = None, image_size: int = GENERATION_IMAGE_SIZE):
        self.cache_dir = Path(cache_dir) if cache_dir else GENERATION_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size

    def _cache_key(self, image_path: Path) -> str:
        digest = hashlib.md5(image_path.read_bytes()).hexdigest()[:16]
        return f"{image_path.stem}_{digest}"

    def extract(self, image_path: Path) -> dict:
        from PIL import Image

        image_path = Path(image_path)
        key = self._cache_key(image_path)
        canny_path = self.cache_dir / f"{key}_canny.png"
        depth_path = self.cache_dir / f"{key}_depth.png"

        if canny_path.exists() and depth_path.exists():
            return {
                "canny": Image.open(canny_path).convert("RGB"),
                "depth": Image.open(depth_path).convert("RGB"),
            }

        img = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        canny_detector, depth_detector = _get_detectors()

        canny_img = canny_detector(img, low_threshold=100, high_threshold=200)
        depth_img = depth_detector(img)

        canny_img.save(canny_path)
        depth_img.save(depth_path)
        return {"canny": canny_img, "depth": depth_img}


# ============================================================
# 3. SDXL + ControlNet(canny+depth) + IP-Adapter 파이프라인 (lazy singleton)
# ============================================================

_pipeline_cache = None


def _load_pipeline():
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

    # 주의: xinsir/controlnet-union-sdxl-1.0 + StableDiffusionXLControlNetUnionPipeline은
    # IP-Adapter를 제대로 지원하지 않아 "'tuple' object has no attribute 'shape'" 에러가 남.
    # canny/depth 각각 공식 개별 ControlNet(-small)으로 분리 + 표준 파이프라인 사용.
    canny_controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_CANNY_MODEL_ID, torch_dtype=torch.float16
    )
    depth_controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_DEPTH_MODEL_ID, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(SDXL_VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE_MODEL_ID,
        controlnet=[canny_controlnet, depth_controlnet],
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO_ID,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT_NAME,
    )
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
    # 이전에 "tuple object has no attribute shape" 에러가 난 건 offload 자체 문제가 아니라
    # StableDiffusionXLControlNetUnionPipeline이 IP-Adapter를 지원 안 해서였음. 표준
    # ControlNet 파이프라인(IP-Adapter 정식 지원)으로 바꿨으니 offload를 다시 사용해서
    # T4 VRAM(14.56GB)에 SDXL+ControlNet 2개+IP-Adapter가 한번에 다 안 올라가는 문제를 해결.
    # (load_ip_adapter 이후에 호출해야 함 — 순서 중요)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()  # VAE 디코딩 피크 메모리 절감 (UNet 어텐션과 무관해 안전)

    _pipeline_cache = pipe
    return pipe


# ============================================================
# 4. 프롬프트 조립
# ============================================================


def _resolve_mood_preset(mood_id: str) -> dict:
    return MOOD_GEN_PRESETS.get(mood_id, DEFAULT_GEN_PRESET)


# 1위-2위 무드 점수차가 이 이상이면 "확실한 매칭"으로 취급 (프리셋 문구를 그대로 적용)
MOOD_MARGIN_CONFIDENT = 0.08
# 매칭이 애매할 때도 controlnet/IP-Adapter 영향력을 완전히 0으로 죽이진 않고 이 비율까지만 낮춤
MOOD_CONFIDENCE_FLOOR = 0.4


def _mood_match_signal(top_moods: list[dict]) -> tuple[float, bool]:
    """1위-2위 무드 점수차(margin)로 매칭 확신도를 계산.
    "베이지·현대식인데 포인트로 파스�텔" 같은 문장은 CLIP이 '파스텔' 단어에 끌려가
    cute_pastel을 1위로 뽑아버리는데, margin이 크지 않으면(=애매한 매칭이면) 그 무드의
    프리셋/레퍼런스 영향력을 낮춰서 사용자가 실제로 쓴 프롬프트 쪽 비중을 높인다."""
    if len(top_moods) < 2:
        return 1.0, True
    margin = top_moods[0]["score"] - top_moods[1]["score"]
    is_confident = margin >= MOOD_MARGIN_CONFIDENT
    frac = max(0.0, min(1.0, margin / MOOD_MARGIN_CONFIDENT))
    confidence = MOOD_CONFIDENCE_FLOOR + (1 - MOOD_CONFIDENCE_FLOOR) * frac
    return confidence, is_confident


def build_generation_prompt(
    prompt_en: str, mood_id: str, room_guard: dict, include_mood_preset: bool = True,
) -> tuple[str, str]:
    preset = _resolve_mood_preset(mood_id)
    # 매칭이 애매할 때(include_mood_preset=False)는 무드 프리셋 문구를 아예 빼서
    # "soft pastel colors, plush textures..." 같은 문구가 사용자 프롬프트를 덮어쓰지 않게 한다.
    # (일반 텍스트 프롬프트라 세기 조절이 안 되니 on/off로 처리)
    preset_suffix = preset["prompt_suffix"] if include_mood_preset else ""
    positive = ", ".join(
        p for p in [
            prompt_en,
            preset_suffix,
            room_guard["prompt_suffix"],
            "photorealistic interior photo, 4k, professional real estate photography",
        ] if p
    )
    negative = ", ".join(p for p in [BASE_NEGATIVE_PROMPT, room_guard["negative_suffix"]] if p)
    return positive, negative


# ============================================================
# 5. 후보 생성 (기본 4장, 서로 다른 레퍼런스로 구조 다양성 확보)
# ============================================================


def _select_reference_images(
    prompt_for_search: str,
    mood_ids: list[str],
    num_candidates: int,
    exclude_paths: set[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    # 후보 수보다 넉넉히 뽑아서, 4장이 전부 같은 사진에서 나오지 않도록
    # 서로 다른 레퍼런스 이미지를 하나씩 배정한다 (구조 다양성 확보).
    # top-N 무드를 합친 풀에서, 유사도 순 고정 픽 대신 점수 가중 랜덤 샘플링을 써서
    # 비슷한 프롬프트를 반복 요청해도 매번 똑같은 레퍼런스 사진만 나오지 않게 한다.
    exclude_paths = exclude_paths or set()
    pool = search_images_within_moods(
        prompt_for_search, mood_ids, top_k=max(num_candidates * 4, 12),
        translate_ko=False,
    )
    pool = [item for item in pool if item["path"] not in exclude_paths]

    if not pool:
        # 전부 제외됐으면(재생성 반복) 제외 없이 다시 조회
        pool = search_images_within_moods(
            prompt_for_search, mood_ids, top_k=max(num_candidates * 4, 12),
            translate_ko=False,
        )

    if not pool:
        raise ValueError(f"무드 {mood_ids}에 레퍼런스 이미지가 없습니다.")

    rng = rng or np.random.default_rng()
    scores = np.array([item["score"] for item in pool])
    weights = np.exp((scores - scores.max()) / 0.1)  # softmax(temperature=0.1), 고득점에 가중치
    weights = weights / weights.sum()

    k = min(num_candidates, len(pool))
    idx = rng.choice(len(pool), size=k, replace=False, p=weights)
    refs = [pool[i] for i in idx]
    # num_candidates보다 pool이 작으면 순환시켜서라도 채움
    while len(refs) < num_candidates:
        refs.append(pool[len(refs) % len(pool)])
    return refs


def generate_candidates(
    prompt: str,
    width_m: float,
    depth_m: float,
    height_m: float,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    mood_id: str | None = None,
    translate_ko: bool = True,
    seed_base: int | None = None,
    exclude_ref_paths: set[str] | None = None,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
) -> dict:
    """텍스트+방크기 → 후보 이미지 num_candidates장. mood_id 생략 시 프롬프트로 자동 매칭."""
    import torch

    prepared = prepare_prompt_for_search(prompt, translate_ko=translate_ko)
    prompt_en = prepared["prompt_for_search"]

    if mood_id is None:
        # 스타일/프리셋은 top-1 무드로 고정하되, 레퍼런스 사진 풀은 top-N 무드를 합쳐서
        # 넓힌다 (top-1만 쓰면 프롬프트 디테일이 달라도 항상 같은 좁은 풀에서만 고르게 됨)
        top_moods = search_mood_by_prompt(
            prompt, top_k=MOOD_POOL_TOP_K, translate_ko=False, prepared=prepared,
        )
        if not top_moods:
            raise ValueError("일치하는 무드를 찾지 못했습니다.")
        mood_id = top_moods[0]["mood_id"]
        mood_folder = top_moods[0]["id"]
        mood_folders = [m["id"] for m in top_moods]
        mood_confidence, mood_confident = _mood_match_signal(top_moods)
    else:
        mood_folder = mood_id
        mood_folders = [mood_folder]  # 무드를 직접 지정한 경우엔 그 무드로만 좁힘
        top_moods = []
        mood_confidence, mood_confident = 1.0, True  # 직접 지정했으니 전체 강도로 적용

    room_guard = room_scale_guard(width_m, depth_m, height_m)
    positive_prompt, negative_prompt = build_generation_prompt(
        prompt_en, mood_id, room_guard, include_mood_preset=mood_confident,
    )
    preset = _resolve_mood_preset(mood_id)
    # 매칭이 애매하면 프리셋의 controlnet/IP-Adapter 강도도 같이 낮춰서, 확신 없는 무드의
    # 레퍼런스 사진이 구조·색감을 과하게 강요하지 않게 한다.
    controlnet_scale = [round(s * mood_confidence, 3) for s in preset["controlnet_scale"]]
    ip_adapter_scale = round(IP_ADAPTER_SCALE * mood_confidence, 3)

    refs = _select_reference_images(prompt_en, mood_folders, num_candidates, exclude_ref_paths)

    # 1단계: canny/depth 가이드를 전부 먼저 뽑고 전처리기(Midas 등)를 메모리에서 내림.
    # 전처리기 + SDXL 파이프라인을 동시에 CPU에 들고 있으면 Colab 무료 T4에서 RAM이 터짐.
    extractor = GuideExtractor()
    guides_by_ref = [extractor.extract(MOOD_LIBRARY_DIR / ref["path"]) for ref in refs]
    _free_detectors()

    # 2단계: 그제야 SDXL+ControlNet(canny+depth)+IP-Adapter 파이프라인을 로드해서 순차 생성
    pipe = _load_pipeline()
    # 파이프라인 로드시 고정으로 설정된 IP_ADAPTER_SCALE 대신, 이번 호출의 무드 확신도로
    # 조정된 값을 적용 (diffusers는 호출 사이에 다시 설정하는 걸 허용)
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    from PIL import Image

    candidates = []
    for i, (ref, guides) in enumerate(zip(refs, guides_by_ref)):
        ref_path = MOOD_LIBRARY_DIR / ref["path"]

        if seed_base is not None:
            seed = seed_base + i
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.Generator()
            seed = generator.seed()  # OS 랜덤에서 시드 뽑고 그 값을 그대로 기록 (재현용)

        result_image = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=[guides["canny"], guides["depth"]],
            controlnet_conditioning_scale=controlnet_scale,
            ip_adapter_image=Image.open(ref_path).convert("RGB"),
            guidance_scale=preset["guidance_scale"],
            num_inference_steps=num_inference_steps,
            height=GENERATION_IMAGE_SIZE,
            width=GENERATION_IMAGE_SIZE,
            generator=generator,
        ).images[0]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        candidates.append({
            "index": i,
            "image": result_image,
            "seed": seed,
            "reference_path": ref["path"],
        })

    result = {
        "prompt_original": prepared["prompt_original"],
        "prompt_en": prompt_en,
        "mood_id": mood_id,
        "mood_folder": mood_folder,
        "mood_scores": top_moods,  # 자동 매칭이었을 때 top-N 무드·점수 (직접 지정 시 빈 리스트)
        "mood_confidence": mood_confidence,  # 1.0=확실한 매칭, 낮을수록 프리셋/레퍼런스 영향력을 줄임
        "room": room_guard,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "used_reference_paths": [r["path"] for r in refs],
        "candidates": candidates,
    }
    return result


def refine_candidates(
    prev_result: dict,
    feedback: str,
    num_candidates: int | None = None,
    seed_base: int | None = None,
) -> dict:
    """이전 결과에 마음에 드는 게 없을 때, 추가 텍스트로 같은 무드 안에서 재생성.
    이전에 썼던 레퍼런스 이미지는 제외해서 매번 새로운 4장이 나오게 한다."""
    combined_prompt = f"{prev_result['prompt_original']}. {feedback}".strip()
    room = prev_result["room"]
    exclude = set(prev_result.get("used_reference_paths") or [])

    return generate_candidates(
        combined_prompt,
        width_m=room["width_m"],
        depth_m=room["depth_m"],
        height_m=room["height_m"],
        num_candidates=num_candidates or len(prev_result["candidates"]),
        mood_id=prev_result["mood_id"],
        translate_ko=True,
        seed_base=seed_base,
        exclude_ref_paths=exclude,
    )


# ============================================================
# 6. 저장 + 표시
# ============================================================


def save_candidates(result: dict, output_dir: Path | None = None) -> dict:
    # candidates[i]["image"](PIL) → 파일 저장, image 키를 image_path로 치환한 JSON 직렬화 가능한 dict 반환
    out_dir = Path(output_dir) if output_dir else GENERATION_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = dict(result)
    saved_candidates = []
    for cand in result["candidates"]:
        filename = f"{result['mood_id']}_{cand['seed']}.png"
        path = out_dir / filename
        cand["image"].save(path)
        saved_candidates.append({
            "index": cand["index"],
            "seed": cand["seed"],
            "reference_path": cand["reference_path"],
            "image_path": str(path),
        })
    saved["candidates"] = saved_candidates

    meta_path = out_dir / f"{result['mood_id']}_{result['candidates'][0]['seed']}_meta.json"
    meta_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["meta_path"] = str(meta_path)
    return saved


def plot_candidates(result: dict, prompt: str = "") -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if font_name in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False

    candidates = result["candidates"]
    fig, axes = plt.subplots(1, len(candidates), figsize=(4 * len(candidates), 4.5))
    if len(candidates) == 1:
        axes = [axes]

    for ax, cand in zip(axes, candidates):
        img = cand["image"] if "image" in cand else None
        if img is None and "image_path" in cand:
            from PIL import Image

            img = Image.open(cand["image_path"])
        ax.imshow(img)
        ax.set_title(f"#{cand['index']} seed={cand['seed']}", fontsize=10)
        ax.axis("off")

    title_suffix = f' — "{prompt[:50]}"' if prompt else ""
    fig.suptitle(
        f"{result['mood_id']} · {result['room']['size_class']} ({result['room']['floor_area_m2']}㎡){title_suffix}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()
