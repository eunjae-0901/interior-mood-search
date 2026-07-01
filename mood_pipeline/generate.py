# 사용자 텍스트 + 방 크기(WxDxH) → SDXL+ControlNetUnion+IP-Adapter로
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

import hashlib
import json

from .config import (
    BASE_NEGATIVE_PROMPT,
    CONTROLNET_MODE_CANNY,
    CONTROLNET_MODE_DEPTH,
    CONTROLNET_UNION_MODEL_ID,
    DEFAULT_GEN_PRESET,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_NUM_CANDIDATES,
    DEPTH_DETECTOR_MODEL_ID,
    GENERATION_CACHE_DIR,
    GENERATION_IMAGE_SIZE,
    GENERATION_OUTPUT_DIR,
    HIGH_CEILING_HEIGHT_M,
    IP_ADAPTER_REPO_ID,
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
    search_images_within_mood,
    search_mood_by_prompt,
)

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
# 3. SDXL + ControlNetUnion + IP-Adapter 파이프라인 (lazy singleton)
# ============================================================

_pipeline_cache = None


def _load_pipeline():
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetUnionModel,
        StableDiffusionXLControlNetUnionPipeline,
    )

    controlnet = ControlNetUnionModel.from_pretrained(
        CONTROLNET_UNION_MODEL_ID, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(SDXL_VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        SDXL_BASE_MODEL_ID,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO_ID,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT_NAME,
    )
    pipe.set_ip_adapter_scale(0.5)
    pipe.enable_model_cpu_offload()  # T4 16GB 대응 (VRAM 절약, 속도는 다소 느려짐)

    _pipeline_cache = pipe
    return pipe


# ============================================================
# 4. 프롬프트 조립
# ============================================================


def _resolve_mood_preset(mood_id: str) -> dict:
    return MOOD_GEN_PRESETS.get(mood_id, DEFAULT_GEN_PRESET)


def build_generation_prompt(prompt_en: str, mood_id: str, room_guard: dict) -> tuple[str, str]:
    preset = _resolve_mood_preset(mood_id)
    positive = ", ".join(
        p for p in [
            prompt_en,
            preset["prompt_suffix"],
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
    mood_id: str,
    num_candidates: int,
    exclude_paths: set[str] | None = None,
) -> list[dict]:
    # 후보 수보다 넉넉히 뽑아서, 4장이 전부 같은 사진에서 나오지 않도록
    # 서로 다른 레퍼런스 이미지를 하나씩 배정한다 (구조 다양성 확보)
    exclude_paths = exclude_paths or set()
    pool = search_images_within_mood(
        prompt_for_search, mood_id=mood_id, top_k=max(num_candidates * 3, 8),
        translate_ko=False,
    )
    pool = [item for item in pool if item["path"] not in exclude_paths]

    if not pool:
        # 전부 제외됐으면(재생성 반복) 제외 없이 다시 조회
        pool = search_images_within_mood(
            prompt_for_search, mood_id=mood_id, top_k=max(num_candidates * 3, 8),
            translate_ko=False,
        )

    if not pool:
        raise ValueError(f"무드 '{mood_id}'에 레퍼런스 이미지가 없습니다.")

    # num_candidates보다 pool이 작으면 순환시켜서라도 채움
    refs = [pool[i % len(pool)] for i in range(num_candidates)]
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
        top_moods = search_mood_by_prompt(prompt, top_k=1, translate_ko=False, prepared=prepared)
        if not top_moods:
            raise ValueError("일치하는 무드를 찾지 못했습니다.")
        mood_id = top_moods[0]["mood_id"]
        mood_folder = top_moods[0]["id"]
    else:
        mood_folder = mood_id

    room_guard = room_scale_guard(width_m, depth_m, height_m)
    positive_prompt, negative_prompt = build_generation_prompt(prompt_en, mood_id, room_guard)
    preset = _resolve_mood_preset(mood_id)

    refs = _select_reference_images(prompt_en, mood_folder, num_candidates, exclude_ref_paths)

    pipe = _load_pipeline()
    extractor = GuideExtractor()

    candidates = []
    for i, ref in enumerate(refs):
        ref_path = MOOD_LIBRARY_DIR / ref["path"]
        guides = extractor.extract(ref_path)

        if seed_base is not None:
            seed = seed_base + i
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.Generator()
            seed = generator.seed()  # OS 랜덤에서 시드 뽑고 그 값을 그대로 기록 (재현용)

        from PIL import Image

        result_image = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            control_image=[guides["canny"], guides["depth"]],
            control_mode=[CONTROLNET_MODE_CANNY, CONTROLNET_MODE_DEPTH],
            controlnet_conditioning_scale=preset["controlnet_scale"],
            ip_adapter_image=Image.open(ref_path).convert("RGB"),
            guidance_scale=preset["guidance_scale"],
            num_inference_steps=num_inference_steps,
            height=GENERATION_IMAGE_SIZE,
            width=GENERATION_IMAGE_SIZE,
            generator=generator,
        ).images[0]

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
