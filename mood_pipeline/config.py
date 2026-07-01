# 무드 라이브러리 파이프라인 공통 설정
from pathlib import Path  # 파일·폴더 경로 처리

# 프로젝트 루트 (mood_pipeline/ 의 상위 폴더)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 클러스터링 대상 원본 이미지 폴더
IMAGE_ROOT = PROJECT_ROOT / "images" / "final"
# 임베딩·클러스터·메타데이터 저장 폴더
DATA_DIR = PROJECT_ROOT / "data"
# 최종 무드 라이브러리 출력 폴더
MOOD_LIBRARY_DIR = PROJECT_ROOT / "mood_library"
# UMAP·elbow 등 시각화 저장 폴더
PREVIEW_DIR = DATA_DIR / "previews"
# 클러스터별 몽타주 미리보기 저장 폴더
CLUSTER_PREVIEW_DIR = DATA_DIR / "cluster_previews"

# 프롬프트 무드 검색용 CLIP 텍스트·대표 이미지 임베딩 캐시
MOOD_TEXT_EMBEDDINGS_PATH = MOOD_LIBRARY_DIR / "mood_text_embeddings.npy"
MOOD_IDS_PATH = MOOD_LIBRARY_DIR / "mood_ids.json"
MOOD_REP_IMAGE_EMBEDDINGS_PATH = MOOD_LIBRARY_DIR / "mood_rep_image_embeddings.npy"
MOOD_REP_IMAGES_PATH = MOOD_LIBRARY_DIR / "mood_rep_images.json"
# gallery 전체 검색 인덱스 버전 (올리면 자동 재빌드)
MOOD_SEARCH_INDEX_VERSION_PATH = MOOD_LIBRARY_DIR / "search_index_version.txt"
SEARCH_INDEX_VERSION = 2

# Hugging Face CLIP 모델 ID
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# 허용 이미지 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# 최소 이미지 한 변 길이 (px)
MIN_IMAGE_SIZE = 200
# 코사인 유사도가 이 값 이상이면 중복으로 간주
DEDUP_SIMILARITY = 0.95
# CLIP 배치 추론 크기
BATCH_SIZE = 16


def default_n_clusters(n_images: int) -> int:
    # 이미지 15장당 클러스터 1개, 최소 4·최대 15
    per_cluster = n_images // 15 or 4
    return max(4, min(15, per_cluster))


# CLIP zero-shot 분류용 영문 무드 후보 문장
MOOD_CANDIDATES = [
    "warm cozy studio apartment interior",
    "minimal white scandinavian room",
    "modern grey industrial studio",
    "natural wood tone small room",
    "soft beige feminine bedroom",
    "dark moody compact apartment",
    "bright airy open studio",
    "vintage retro small room",
    "monochrome minimalist interior",
    "plant-filled green interior",
    "luxury modern studio apartment",
    "cute pastel room decor",
]

# 영문 무드 → 한글 표시명 매핑
MOOD_KO = {
    "warm cozy studio apartment interior": "따뜻한 코지 원룸",
    "minimal white scandinavian room": "미니멀 화이트",
    "modern grey industrial studio": "모던 그레이",
    "natural wood tone small room": "내추럴 우드",
    "soft beige feminine bedroom": "소프트 베이지",
    "dark moody compact apartment": "다크 무디",
    "bright airy open studio": "밝은 에어리",
    "vintage retro small room": "빈티지 레트로",
    "monochrome minimalist interior": "모노톤 미니멀",
    "plant-filled green interior": "플랜트 그린",
    "luxury modern studio apartment": "럭셔리 모던",
    "cute pastel room decor": "파스텔 귀여운",
}

# 영문 무드 → 폴더명용 slug 매핑
SLUG_MAP = {
    "warm cozy studio apartment interior": "warm_cozy",
    "minimal white scandinavian room": "minimal_white",
    "modern grey industrial studio": "modern_grey",
    "natural wood tone small room": "natural_wood",
    "soft beige feminine bedroom": "soft_beige",
    "dark moody compact apartment": "dark_moody",
    "bright airy open studio": "bright_airy",
    "vintage retro small room": "vintage_retro",
    "monochrome minimalist interior": "monochrome_minimal",
    "plant-filled green interior": "plant_green",
    "luxury modern studio apartment": "luxury_modern",
    "cute pastel room decor": "cute_pastel",
}

# ============================================================
# Model 1 이미지 생성 (SDXL + ControlNet(canny+depth) + IP-Adapter)
# ============================================================
# Colab GPU 전제 — 로컬 CPU 환경에서는 import만 되고 실제 파이프라인은 안 돌아감

# SDXL 베이스 모델
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# fp16 SDXL과 궁합이 맞는 VAE (기본 VAE는 fp16에서 NaN 나는 이슈가 있어 대체)
SDXL_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
# 주의: xinsir/controlnet-union-sdxl-1.0 + StableDiffusionXLControlNetUnionPipeline 조합은
# IP-Adapter를 제대로 지원하지 않아 "'tuple' object has no attribute 'shape'" 에러가 남
# (diffusers에서 IP-Adapter는 표준 StableDiffusionXLControlNetPipeline 위주로 지원됨).
# 그래서 canny/depth를 각각의 공식 개별 ControlNet(-small, 가벼움)으로 분리해서 사용.
CONTROLNET_CANNY_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0-small"
CONTROLNET_DEPTH_MODEL_ID = "diffusers/controlnet-depth-sdxl-1.0-small"
IP_ADAPTER_REPO_ID = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sdxl.bin"
# 0.5는 레퍼런스 사진의 식물/장식 같은 요소까지 그대로 따라와서 텍스트 프롬프트와 안 맞는
# 물건이 끼어드는 경향이 있어 낮춤. 스타일/톤은 유지하되 내용은 텍스트가 주도하게.
IP_ADAPTER_SCALE = 0.3
# Depth 전처리 모델 (controlnet_aux)
DEPTH_DETECTOR_MODEL_ID = "lllyasviel/Annotators"

# 가이드(canny/depth) 이미지 캐시 폴더 — Colab에서는 config 임포트 후
# mood_pipeline.config.GENERATION_CACHE_DIR = Path("/content/drive/MyDrive/.../guides") 로 덮어써서 Drive에 영구 저장
GENERATION_CACHE_DIR = DATA_DIR / "generation_cache" / "guides"
# 생성된 후보 이미지 + 메타데이터 저장 폴더 (역시 Colab에서 Drive 경로로 덮어쓰기 권장)
GENERATION_OUTPUT_DIR = DATA_DIR / "generation_cache" / "outputs"

# T4(14.56GB 가용 VRAM) 기준: SDXL+ControlNetUnion+IP-Adapter 가중치만 fp16으로
# ~14GB를 써서 1024 해상도는 활성화 메모리 여유가 거의 없어(OOM) 768로 낮춤
GENERATION_IMAGE_SIZE = 768
DEFAULT_NUM_CANDIDATES = 4
DEFAULT_INFERENCE_STEPS = 30

BASE_NEGATIVE_PROMPT = (
    "blurry, cartoon, illustration, distorted, low quality, oversaturated, "
    "watermark, text, logo, extra walls, warped perspective, "
    "cluttered, mismatched furniture, illogical furniture placement, "
    "floating furniture, furniture overlapping, impossible room layout, "
    "random unrelated objects, extra plants, extra decorations not mentioned"
)

# 무드별 생성 프리셋 — controlnet_scale: [canny(구조), depth(공간감)]
MOOD_GEN_PRESETS = {
    "warm_cozy": {
        "controlnet_scale": [0.55, 0.5],
        "guidance_scale": 6.5,
        "prompt_suffix": "warm lighting, wooden textures, cozy atmosphere",
    },
    "minimal_white": {
        "controlnet_scale": [0.7, 0.55],
        "guidance_scale": 7.5,
        "prompt_suffix": "clean lines, white walls, minimal furniture",
    },
    "modern_grey": {
        "controlnet_scale": [0.65, 0.55],
        "guidance_scale": 7.0,
        "prompt_suffix": "industrial grey tones, concrete texture, modern fixtures",
    },
    "natural_wood": {
        "controlnet_scale": [0.6, 0.5],
        "guidance_scale": 6.5,
        "prompt_suffix": "natural wood furniture, warm neutral tones, plants",
    },
    "soft_beige": {
        "controlnet_scale": [0.55, 0.45],
        "guidance_scale": 6.5,
        "prompt_suffix": "soft beige palette, feminine textures, gentle lighting",
    },
    "dark_moody": {
        "controlnet_scale": [0.65, 0.55],
        "guidance_scale": 7.5,
        "prompt_suffix": "dark moody tones, low-key lighting, matte finishes",
    },
    "bright_airy": {
        "controlnet_scale": [0.55, 0.45],
        "guidance_scale": 6.5,
        "prompt_suffix": "bright natural light, airy open feel, light tones",
    },
    "vintage_retro": {
        "controlnet_scale": [0.6, 0.5],
        "guidance_scale": 7.0,
        "prompt_suffix": "vintage retro furniture, warm film-like tones",
    },
    "monochrome_minimal": {
        "controlnet_scale": [0.7, 0.55],
        "guidance_scale": 7.5,
        "prompt_suffix": "monochrome palette, minimal decor, sharp lines",
    },
    "plant_green": {
        "controlnet_scale": [0.55, 0.5],
        "guidance_scale": 6.5,
        "prompt_suffix": "lush indoor plants, green accents, natural light",
    },
    "luxury_modern": {
        "controlnet_scale": [0.65, 0.55],
        "guidance_scale": 8.0,
        "prompt_suffix": "high-end materials, marble, gold accents",
    },
    "cute_pastel": {
        "controlnet_scale": [0.5, 0.4],
        "guidance_scale": 6.0,
        "prompt_suffix": "soft pastel colors, plush textures, playful decor",
    },
}
# 프리셋 없는 무드용 기본값 (MOOD_GEN_PRESETS에 키가 없을 때)
DEFAULT_GEN_PRESET = {
    "controlnet_scale": [0.6, 0.5],
    "guidance_scale": 7.0,
    "prompt_suffix": "",
}

# 방 크기(㎡) 구간별 스케일 가드 — "5x7 원룸에 대저택 거실"처럼
# 실제 방 크기와 안 맞는 비현실적 스케일이 나오지 않도록 프롬프트에 강제 주입
ROOM_SIZE_BRACKETS = [
    # (최대 바닥면적㎡, size_class, prompt_suffix, negative_suffix)
    (10, "tiny_studio", "very small compact studio apartment room, tight space, single bed or small sofa scale furniture",
     "spacious room, large hall, high ceiling, mansion, villa, oversized furniture, grand living room"),
    (20, "small_studio", "small one-room studio apartment, compact human-scale furniture",
     "spacious room, large hall, mansion, villa, oversized furniture, grand living room"),
    (35, "mid_room", "modest one-room apartment, standard-scale furniture",
     "mansion, villa, ballroom, palatial interior, oversized furniture"),
    (60, "spacious_room", "moderately spacious one-room apartment",
     "mansion, palatial interior, ballroom"),
]
ROOM_SIZE_FALLBACK = (
    "large open-plan apartment",
    "palatial mansion interior, ballroom",
)
LOW_CEILING_HEIGHT_M = 2.3  # 이 미만이면 "낮은 천장" 문구 추가
HIGH_CEILING_HEIGHT_M = 2.8  # 이 초과면 "높은 천장" 허용 문구 추가
