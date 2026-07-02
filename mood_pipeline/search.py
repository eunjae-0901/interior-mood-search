# 프롬프트 → CLIP 텍스트 임베딩 → mood_library 코사인 유사도 검색
from __future__ import annotations

import sys
from pathlib import Path

# IDE에서 search.py 를 직접 실행할 때 상대 import 오류 방지
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "mood_pipeline"

import hashlib
import json

import numpy as np

from .config import (
    MOOD_IDS_PATH,
    MOOD_LIBRARY_DIR,
    MOOD_REP_IMAGE_EMBEDDINGS_PATH,
    MOOD_REP_IMAGES_PATH,
    MOOD_SEARCH_INDEX_VERSION_PATH,
    MOOD_TEXT_EMBEDDINGS_PATH,
    SEARCH_INDEX_VERSION,
    IMAGE_EXTENSIONS,
)
from .embed import encode_images, encode_texts, load_clip

# CLIP 모델 재로드 방지 (검색 반복 호출 시)
_clip_cache: tuple | None = None


def _get_clip():
    global _clip_cache
    if _clip_cache is None:
        _clip_cache = load_clip()
    return _clip_cache


def _load_index() -> dict:
    index_path = MOOD_LIBRARY_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"mood_library/index.json 없음: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _load_meta(folder_id: str) -> dict:
    meta_path = MOOD_LIBRARY_DIR / folder_id / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_mood_search_text(index_entry: dict, meta: dict) -> str:
    # index + meta 필드를 하나의 검색용 문장으로 합침.
    # CLIP(openai/clip-vit-base-patch32)은 영어 전용 모델이라 name_ko/description
    # 같은 한글 필드를 섞으면 노이즈 토큰이 되어 매칭 정확도가 떨어진다 → 영어 필드만 사용
    keywords = index_entry.get("keywords") or meta.get("tags") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    typical = meta.get("typical_elements") or []
    if isinstance(typical, str):
        typical = [typical]

    parts = [
        index_entry.get("name_en", ""),
        ", ".join(keywords),
        ", ".join(typical),
        ", ".join(meta.get("tags") or []),
    ]
    # 빈 문자열 제거 후 연결
    return " | ".join(p.strip() for p in parts if p and str(p).strip())


def _path_content_hash(path: Path) -> str:
    # cover 복사본과 gallery 원본 등 동일 이미지 중복 제거용
    return hashlib.md5(path.read_bytes()).hexdigest()


def _collect_rep_image_paths(folder_id: str, index_entry: dict) -> list[Path]:
    # cover + gallery 전체를 검색 후보로 (중복 픽셀은 hash로 1장만)
    mood_dir = MOOD_LIBRARY_DIR / folder_id
    paths: list[Path] = []
    seen_hashes: set[str] = set()

    def add(path: Path):
        if not path.exists():
            return
        digest = _path_content_hash(path)
        if digest in seen_hashes:
            return
        seen_hashes.add(digest)
        paths.append(path)

    add(mood_dir / "cover.jpg")

    gallery = mood_dir / "gallery"
    if not gallery.is_dir():
        return paths

    rep_names = index_entry.get("representatives") or []
    seen_names: set[str] = set()

    for name in rep_names:
        add(gallery / name)
        seen_names.add(name)

    for p in sorted(gallery.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if p.name in seen_names:
            continue
        add(p)

    return paths


def _search_index_up_to_date() -> bool:
    if not MOOD_SEARCH_INDEX_VERSION_PATH.exists():
        return False
    return MOOD_SEARCH_INDEX_VERSION_PATH.read_text(encoding="utf-8").strip() == str(
        SEARCH_INDEX_VERSION
    )


def _dedupe_image_results(results: list[dict], top_k: int) -> list[dict]:
    # 2차 검색: 동일 픽셀(cover 복사본 등) 결과 제거
    seen_hashes: set[str] = set()
    unique: list[dict] = []
    for item in results:
        img_path = MOOD_LIBRARY_DIR / item["path"]
        if not img_path.exists():
            continue
        digest = _path_content_hash(img_path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def build_mood_text_embeddings(force: bool = False) -> dict:
    # mood_library/index.json + meta.json → CLIP 텍스트·대표 이미지 임베딩 저장
    index = _load_index()
    moods = index.get("moods") or []
    if not moods:
        raise ValueError("index.json에 moods가 비어 있습니다.")

    if (
        not force
        and _search_index_up_to_date()
        and MOOD_TEXT_EMBEDDINGS_PATH.exists()
        and MOOD_IDS_PATH.exists()
        and MOOD_REP_IMAGE_EMBEDDINGS_PATH.exists()
        and MOOD_REP_IMAGES_PATH.exists()
    ):
        mood_ids = json.loads(MOOD_IDS_PATH.read_text(encoding="utf-8"))
        rep_images = json.loads(MOOD_REP_IMAGES_PATH.read_text(encoding="utf-8"))
        return {
            "mood_count": len(mood_ids),
            "text_embeddings_shape": tuple(np.load(MOOD_TEXT_EMBEDDINGS_PATH).shape),
            "rep_image_count": len(rep_images),
            "cached": True,
        }

    model, processor, device = _get_clip()

    mood_records = []
    mood_texts = []
    for entry in moods:
        folder_id = entry["id"]
        meta = _load_meta(folder_id)
        text = build_mood_search_text(entry, meta)
        mood_texts.append(text)
        mood_records.append({
            "id": folder_id,
            "mood_id": entry.get("mood_id", folder_id),
            "mood_name_ko": entry.get("name_ko", ""),
            "mood_name_en": entry.get("name_en", ""),
            "search_text": text,
        })

    text_emb = encode_texts(mood_texts, model, processor, device)

    rep_image_records = []
    rep_image_paths: list[Path] = []
    for entry, record in zip(moods, mood_records):
        folder_id = entry["id"]
        for img_path in _collect_rep_image_paths(folder_id, entry):
            rep_image_records.append({
                "id": folder_id,
                "mood_id": record["mood_id"],
                "mood_name_ko": record["mood_name_ko"],
                "filename": img_path.name,
                "path": str(img_path.relative_to(MOOD_LIBRARY_DIR)).replace("\\", "/"),
                "embedding_index": len(rep_image_paths),
            })
            rep_image_paths.append(img_path)

    if rep_image_paths:
        rep_image_emb = encode_images(rep_image_paths, model, processor, device)
    else:
        rep_image_emb = np.empty((0, text_emb.shape[1]), dtype=np.float32)

    MOOD_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MOOD_TEXT_EMBEDDINGS_PATH, text_emb)
    MOOD_IDS_PATH.write_text(
        json.dumps(mood_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(MOOD_REP_IMAGE_EMBEDDINGS_PATH, rep_image_emb)
    MOOD_REP_IMAGES_PATH.write_text(
        json.dumps(rep_image_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    MOOD_SEARCH_INDEX_VERSION_PATH.write_text(str(SEARCH_INDEX_VERSION), encoding="utf-8")

    return {
        "mood_count": len(mood_records),
        "text_embeddings_shape": text_emb.shape,
        "rep_image_count": len(rep_image_records),
        "cached": False,
        "paths": {
            "mood_text_embeddings": str(MOOD_TEXT_EMBEDDINGS_PATH),
            "mood_ids": str(MOOD_IDS_PATH),
            "mood_rep_image_embeddings": str(MOOD_REP_IMAGE_EMBEDDINGS_PATH),
            "mood_rep_images": str(MOOD_REP_IMAGES_PATH),
        },
    }


def _contains_hangul(text: str) -> bool:
    # 한글(가-힣) 포함 여부
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def _translate_ko_to_en(text: str) -> str:
    # 한국어 → 영어 (CLIP 검색용)
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source="ko", target="en").translate(text)


def prepare_prompt_for_search(prompt: str, translate_ko: bool = True) -> dict:
    # 검색에 쓸 프롬프트 준비 (한국어면 영어로 번역)
    original = prompt.strip()
    if not original:
        return {
            "prompt_original": "",
            "prompt_en": "",
            "prompt_for_search": "",
            "translated": False,
        }

    if translate_ko and _contains_hangul(original):
        try:
            translated = _translate_ko_to_en(original)
            return {
                "prompt_original": original,
                "prompt_en": translated,
                "prompt_for_search": translated,
                "translated": True,
            }
        except Exception as exc:
            print(f"Warning: 번역 실패, 원문 사용 ({exc})")

    return {
        "prompt_original": original,
        "prompt_en": original,
        "prompt_for_search": original,
        "translated": False,
    }


def _load_mood_search_cache() -> tuple[np.ndarray, list[dict]]:
    if not MOOD_TEXT_EMBEDDINGS_PATH.exists() or not MOOD_IDS_PATH.exists():
        build_mood_text_embeddings(force=True)

    text_emb = np.load(MOOD_TEXT_EMBEDDINGS_PATH)
    mood_ids = json.loads(MOOD_IDS_PATH.read_text(encoding="utf-8"))
    return text_emb, mood_ids


def _encode_prompt(prompt: str) -> np.ndarray:
    model, processor, device = _get_clip()
    return encode_texts([prompt], model, processor, device)[0]


def search_mood_by_prompt(
    prompt: str,
    top_k: int = 1,  # ★ 1차: 선택 무드(cover) 개수 — UI 후보 보여줄 때만 3 등으로 변경
    translate_ko: bool = True,
    prepared: dict | None = None,
) -> list[dict]:
    # 1차: 사용자 프롬프트 ↔ 무드 meta 텍스트 임베딩 → plot_search_result 위쪽 figure
    if prepared is None:
        prepared = prepare_prompt_for_search(prompt, translate_ko=translate_ko)
    text_emb, mood_ids = _load_mood_search_cache()
    prompt_emb = _encode_prompt(prepared["prompt_for_search"])

    scores = text_emb @ prompt_emb  # (N,) 코사인 유사도
    top_k = min(top_k, len(scores))
    ranked = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked:
        info = mood_ids[int(idx)]
        results.append({
            "mood_id": info["mood_id"],
            "id": info["id"],
            "mood_name_ko": info["mood_name_ko"],
            "mood_name_en": info.get("mood_name_en", ""),
            "score": float(scores[idx]),
        })
    return results


def search_images_within_moods(
    prompt: str,
    mood_ids: list[str],
    top_k: int = 2,  # ★ 2차: gallery 추천 이미지 개수 (5장 원하면 5)
    translate_ko: bool = True,
    prepared: dict | None = None,
) -> list[dict]:
    # 2차: 무드 1개가 아니라 top-N 무드 gallery를 합쳐서 프롬프트와 재정렬.
    # 무드 하나로만 좁히면 프롬프트 디테일이 달라져도 항상 같은 좁은 풀에서만
    # 고르게 돼서 추천이 비슷해지는 문제가 있어 인접 무드까지 후보에 포함한다.
    if not MOOD_REP_IMAGE_EMBEDDINGS_PATH.exists() or not MOOD_REP_IMAGES_PATH.exists():
        build_mood_text_embeddings(force=True)

    rep_emb = np.load(MOOD_REP_IMAGE_EMBEDDINGS_PATH)
    rep_images = json.loads(MOOD_REP_IMAGES_PATH.read_text(encoding="utf-8"))

    # mood_id 또는 folder id 둘 다 매칭
    mood_id_set = set(mood_ids)
    candidates = [
        (i, rec)
        for i, rec in enumerate(rep_images)
        if rec["mood_id"] in mood_id_set or rec["id"] in mood_id_set
    ]
    if not candidates:
        return []

    if prepared is None:
        prepared = prepare_prompt_for_search(prompt, translate_ko=translate_ko)
    prompt_emb = _encode_prompt(prepared["prompt_for_search"])
    results = []
    for i, rec in candidates:
        emb_idx = rec["embedding_index"]
        score = float(rep_emb[emb_idx] @ prompt_emb)
        results.append({
            "mood_id": rec["mood_id"],
            "id": rec["id"],
            "mood_name_ko": rec["mood_name_ko"],
            "filename": rec["filename"],
            "path": rec["path"],
            "score": score,
        })

    results.sort(key=lambda x: -x["score"])
    return _dedupe_image_results(results, top_k)


def search_images_within_mood(
    prompt: str,
    mood_id: str,
    top_k: int = 2,
    translate_ko: bool = True,
    prepared: dict | None = None,
) -> list[dict]:
    # 무드 1개짜리 검색 (하위 호환용 래퍼)
    return search_images_within_moods(
        prompt, [mood_id], top_k=top_k, translate_ko=translate_ko, prepared=prepared,
    )


def search_mood_with_images(
    prompt: str,
    top_k: int = 2,  # ★ 추천 gallery 이미지 개수 (5장 원하면 top_k=5). 1차 무드 cover는 항상 1개
    mood_top_k: int = 2,  # ★ 2차 풀을 합칠 후보 무드 개수 (1이면 예전처럼 top-1 무드만)
    translate_ko: bool = True,
) -> dict:
    # 노트북/CLI 진입점 — 출력 장수는 top_k 하나만 조절
    prepared = prepare_prompt_for_search(prompt, translate_ko=translate_ko)
    moods = search_mood_by_prompt(
        prompt,
        top_k=mood_top_k,
        translate_ko=translate_ko,
        prepared=prepared,
    )
    if not moods:
        return {
            "prompt_original": prepared["prompt_original"],
            "prompt_en": prepared["prompt_en"],
            "translated": prepared["translated"],
            "moods": [],
            "selected_mood": None,
            "recommended_images": [],
        }

    selected = moods[0]
    images = search_images_within_moods(
        prompt,
        mood_ids=[m["mood_id"] for m in moods],
        top_k=top_k,
        translate_ko=translate_ko,
        prepared=prepared,
    )
    return {
        "prompt_original": prepared["prompt_original"],
        "prompt_en": prepared["prompt_en"],
        "translated": prepared["translated"],
        "moods": moods,
        "selected_mood": selected,
        "recommended_images": images,
    }


def plot_search_result(
    result: dict,
    prompt: str = "",
    library_dir: Path | None = None,
) -> None:
    # search_mood_with_images() 결과를 matplotlib figure로 표시
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    # Windows/macOS 한글 제목용 폰트 (없으면 기본 폰트)
    for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if font_name in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False

    lib = library_dir or MOOD_LIBRARY_DIR
    moods = result.get("moods") or []
    images = result.get("recommended_images") or []

    title_suffix = f' — "{prompt[:50]}"' if prompt else ""

    if moods:
        n_moods = len(moods)
        fig1, axes1 = plt.subplots(1, n_moods, figsize=(4 * n_moods, 4.5))
        if n_moods == 1:
            axes1 = [axes1]
        for ax, mood in zip(axes1, moods):
            cover = lib / mood["id"] / "cover.jpg"
            if cover.exists():
                ax.imshow(Image.open(cover).convert("RGB"))
            else:
                ax.text(0.5, 0.5, "cover 없음", ha="center", va="center")
            ax.set_title(
                f"{mood['mood_name_ko']}\n{mood['mood_id']} · cos={mood['score']:.3f}",
                fontsize=10,
            )
            ax.axis("off")
        fig1.suptitle(f"선택 무드{title_suffix}", fontsize=12)
        fig1.tight_layout()
        plt.show()

    if images:
        cols = min(3, len(images))
        rows = (len(images) + cols - 1) // cols
        fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes2 = np.array([[axes2]])
        elif rows == 1:
            axes2 = axes2.reshape(1, -1)
        elif cols == 1:
            axes2 = axes2.reshape(-1, 1)

        for ax in axes2.flat:
            ax.axis("off")

        for i, item in enumerate(images):
            r, c = divmod(i, cols)
            img_path = lib / item["path"]
            if img_path.exists():
                axes2[r, c].imshow(Image.open(img_path).convert("RGB"))
            axes2[r, c].set_title(
                f"{item['filename'][:16]}\ncos={item['score']:.3f}",
                fontsize=9,
            )
            axes2[r, c].axis("off")

        fig2.suptitle(f"추천 레퍼런스 이미지{title_suffix}", fontsize=12)
        fig2.tight_layout()
        plt.show()
    elif not moods:
        print("표시할 검색 결과가 없습니다.")


def _cli_main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="프롬프트 기반 무드 검색")
    parser.add_argument("prompt", help="검색할 프롬프트 문장")
    parser.add_argument("--top-k", type=int, default=2, help="추천 gallery 이미지 개수")
    parser.add_argument("--moods-only", action="store_true", help="1차 무드 검색만")
    parser.add_argument("--plot", action="store_true", help="matplotlib figure 표시")
    parser.add_argument("--no-translate", action="store_true", help="한→영 번역 비활성화")
    args = parser.parse_args()

    translate_ko = not args.no_translate

    if args.moods_only:
        result = search_mood_by_prompt(
            args.prompt,
            top_k=args.top_k,
            translate_ko=translate_ko,
        )
        if translate_ko:
            prepared = prepare_prompt_for_search(args.prompt, translate_ko=True)
            print(json.dumps({
                "prompt_original": prepared["prompt_original"],
                "prompt_en": prepared["prompt_en"],
                "translated": prepared["translated"],
                "moods": result,
            }, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = search_mood_with_images(
            args.prompt,
            top_k=args.top_k,
            translate_ko=translate_ko,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.plot and isinstance(result, dict):
        plot_search_result(result, prompt=args.prompt)


if __name__ == "__main__":
    _cli_main()
