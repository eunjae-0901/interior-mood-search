# CLIP 클러스터링 → 분류 → 라벨링 → 무드 라이브러리 → 프롬프트 검색 가이드

오늘의집·Pinterest에서 수집한 원룸 인테리어 이미지(`images/final/`)를 기반으로, **시각적 무드(mood)** 단위의 라이브러리를 만들고, **사용자 프롬프트로 무드를 자동 추천**하는 전체 파이프라인입니다.

---

## 1. 목표 정의

| 단계 | 산출물 | 설명 |
|------|--------|------|
| CLIP 임베딩 | `embeddings.npy` | 각 이미지를 512차원 벡터로 표현 |
| 클러스터링 | `clusters.json` | 비슷한 분위기끼리 자동 그룹화 |
| 분류 | `labels.json` | 클러스터 → 무드 카테고리 매핑 |
| 라벨링 | `metadata.csv` | 이미지별 최종 태그·설명 |
| 무드 라이브러리 | `mood_library/` | 카테고리별 대표 이미지 + 메타데이터 |
| **프롬프트 검색** | `mood_text_embeddings.npy` 등 | 사용자 문장 → CLIP → 가장 유사한 무드 자동 선택 |

**최종 사용 예 (변경됨):**

사용자가 무드를 직접 고르는 대신, 아래처럼 **자연어 프롬프트**를 입력합니다.

```text
"따뜻하고 우드톤이 있는 아늑한 원룸"
```

→ CLIP text embedding → `mood_library` 무드 meta embedding과 코사인 유사도 비교 → Top-K 무드 추천 + 해당 무드 내 대표 이미지 추천

---

## 2. 권장 폴더 구조

```
창의학기/
├── images/
│   ├── final/                          # 현재 파이프라인 입력 (통합 이미지)
│   ├── pinterest_원룸_인테리어/         # Pinterest 크롤 결과
│   └── 원룸_인테리어_오늘의집/          # 오늘의집 크롤 결과
├── data/
│   ├── raw/                            # 전처리 전 복사본 (선택)
│   ├── embeddings.npy                  # (N, 512) CLIP 이미지 벡터
│   ├── image_paths.json                # 인덱스 ↔ 파일명 매핑
│   ├── clusters.json                   # 클러스터 ID
│   ├── cluster_labels.npy              # (N,) 클러스터 라벨 배열
│   ├── labels.json                     # 클러스터 → 무드명
│   ├── metadata.csv                    # 이미지별 최종 메타
│   ├── cluster_previews/               # 클러스터 montage PNG
│   └── previews/                       # UMAP·elbow 시각화
├── mood_library/
│   ├── index.json                      # 전체 무드 카탈로그
│   ├── mood_text_embeddings.npy        # 무드 meta 텍스트 CLIP 임베딩 (M, 512)
│   ├── mood_ids.json                   # 무드 메타 + search_text
│   ├── mood_rep_image_embeddings.npy   # 대표 이미지 CLIP 임베딩
│   ├── mood_rep_images.json            # 대표 이미지 경로·mood_id 매핑
│   ├── natural_wood_c00/
│   │   ├── cover.jpg
│   │   ├── gallery/
│   │   └── meta.json
│   └── ...
├── mood_pipeline/
│   ├── config.py
│   ├── preprocess.py
│   ├── embed.py                        # 이미지·텍스트 CLIP 임베딩
│   ├── cluster.py
│   ├── label.py
│   ├── build_library.py
│   └── search.py                       # 프롬프트 기반 무드 검색
├── notebooks/
│   ├── 01_clip_embed.ipynb
│   ├── 02_cluster.ipynb
│   ├── 03_label.ipynb
│   ├── 04_build_library.ipynb
│   └── 05_search_mood.ipynb            # 프롬프트 검색 데모
├── ohouse_crawl.ipynb                   # 오늘의집 크롤러
├── pinterest_board_crawl.ipynb         # Pinterest 보드 크롤러
├── run_pipeline.py                     # 전체 파이프라인 일괄 실행
└── requirements-ml.txt
```

---

## 3. 전체 파이프라인 (6단계)

```
[원본 이미지]  images/final/
     ↓ 0. 전처리·검증
[CLIP 이미지 임베딩]
     ↓ 1. embeddings.npy (512-d)
[클러스터링]
     ↓ 2. cluster_id (KMeans, k 자동/수동)
[CLIP Zero-shot 라벨링]
     ↓ 3. labels.json, metadata.csv (클러스터당 무드 1개, 중복 방지)
[무드 라이브러리 빌드]
     ↓ 4. mood_library/ (cover, gallery, meta.json, index.json)
[프롬프트 검색 인덱스 빌드]
     ↓ 5. mood_text_embeddings.npy, mood_rep_image_embeddings.npy
[런타임: 사용자 프롬프트 입력]
     ↓ 6. search_mood_by_prompt() → Top-K 무드 + 대표 이미지 추천
```

---

## 4. Phase 0 — 데이터 준비

### 4.1 수집량

- **현재:** `images/final/` 약 360~400장 (중복 제거 후 ~361장)
- **권장:** 무드 라이브러리·검색 품질을 위해 **300~500장** (클러스터당 15~30장 이상)
- 그레이·화이트·우드 등 **컬러/톤별로 따로 수집**하면 무드 다양성 확보에 유리

### 4.2 크롤러

| 노트북 | 소스 | 저장 경로 |
|--------|------|-----------|
| `ohouse_crawl.ipynb` | 오늘의집 "모던 인테리어" 피드 | `images/모던_인테리어_오늘의집/` |
| `pinterest_board_crawl.ipynb` | Pinterest 보드 | `images/pinterest_원룸_인테리어/` |

수집 후 `images/final/`로 통합해 파이프라인 입력으로 사용합니다. (`mood_pipeline/config.py` → `IMAGE_ROOT`)

### 4.3 전처리 체크리스트

- [ ] 깨진 파일 제거 (`PIL.Image.open` 실패 건)
- [ ] 너무 작은 이미지 제거 (한 변 < 200px)
- [ ] **시각적 중복** CLIP 코사인 유사도 ≥ 0.95 로 제거 (`deduplicate_by_embedding`)
- [ ] 프로필 사진 등 비인테리어 이미지 수동 제거

---

## 5. Phase 1 — CLIP 임베딩 추출

### 5.1 모델

| 모델 | 장점 | 비고 |
|------|------|------|
| `openai/clip-vit-base-patch32` | 가볍고 빠름, 512-d | **현재 프로젝트 기본값** |
| `openai/clip-vit-large-patch14` | 정확도 높음 | GPU·시간 더 필요 |

### 5.2 실행

```bash
# 노트북
notebooks/01_clip_embed.ipynb

# 또는 모듈
python -c "from mood_pipeline.embed import run_embedding; run_embedding()"
```

### 5.3 산출물

| 파일 | 설명 |
|------|------|
| `data/embeddings.npy` | (N, 512) L2 정규화 이미지 임베딩 |
| `data/image_paths.json` | 임베딩 행 ↔ 상대 경로 |
| `data/image_manifest.json` | count, deduplicated 등 요약 |

### 5.4 주의사항

- 임베딩은 **한 번만** 추출하고 이후 단계에서 재사용
- GPU 없어도 400장은 CPU로 수분 내 가능
- `transformers` v5 API 호환 (`get_image_features`, `_as_tensor` 헬퍼)

---

## 6. Phase 2 — 클러스터링

### 6.1 방법

현재 구현: **KMeans** + silhouette score로 k 자동 선택 (범위 약 8~15)

| 방법 | 상태 | 비고 |
|------|------|------|
| **KMeans** | ✅ 구현됨 | `run_clustering()`, `find_best_k()` |
| HDBSCAN | 미구현 | 필요 시 확장 |
| UMAP | ✅ 시각화용 | `plot_umap()` |

### 6.2 실행

```bash
notebooks/02_cluster.ipynb
# N_CLUSTERS = None  → 자동, 또는 8~15 직접 지정
```

### 6.3 산출물

| 파일 | 설명 |
|------|------|
| `data/clusters.json` | 파일명 → cluster_id |
| `data/cluster_labels.npy` | (N,) 클러스터 ID 배열 |
| `data/cluster_summary.json` | k, cluster_sizes, silhouette |
| `data/previews/umap_clusters.png` | UMAP 시각화 |

---

## 7. Phase 3 — 분류 (CLIP Zero-shot)

클러스터 centroid와 `MOOD_CANDIDATES` 텍스트 임베딩의 코사인 유사도로 무드를 배정합니다.

### 7.1 중복 라벨 방지

원룸 이미지는 CLIP상 대부분 "warm cozy"에 높은 점수를 받을 수 있어, **클라이언트마다 독립 argmax**를 하면 전부 같은 무드가 됩니다.

→ `label.py`에서 **확신도(1위−2위)가 큰 클러스터부터** 무드 라벨을 선점하고, **클러스터마다 서로 다른 무드 1개**를 배정합니다.

### 7.2 무드 후보 (`config.py`)

12개 영문 문장 + `MOOD_KO` 한글 매핑 + `SLUG_MAP` 폴더 slug.  
예: `natural_wood` → 내추럴 우드, `minimal_white` → 미니멀 화이트

### 7.3 실행

```bash
notebooks/03_label.ipynb
```

산출물: `labels.json`, `metadata.csv`, `data/cluster_previews/*.png`

---

## 8. Phase 4 — 라벨링 (자동 + 수동 검수)

### 8.1 워크플로

1. `03_label.ipynb` 실행 → 클러스터별 montage 생성
2. `data/cluster_previews/`에서 무드명·이미지 그룹 확인
3. 틀린 클러스터는 `data/labels.json` 수동 수정
4. `04_build_library.ipynb`만 다시 실행 (임베딩·클러스터 재실행 불필요)

### 8.2 `metadata.csv` 주요 컬럼

| 컬럼 | 설명 |
|------|------|
| `filename` | 파일명 |
| `cluster_id` | 클러스터 번호 |
| `mood_id` | slug (예: `warm_cozy`) |
| `mood_name_ko` | 한글 표시명 |
| `is_representative` | 대표 이미지 여부 |
| `tags` | mood_id 기반 태그 |

---

## 9. Phase 5 — 무드 라이브러리 구축

### 9.1 폴더 구조 (무드 1개 예시)

```
mood_library/natural_wood_c00/
├── cover.jpg           # 대표 1장
├── gallery/            # 해당 무드 전체 이미지
└── meta.json           # name_ko, tags, color_palette, confidence
```

### 9.2 `index.json`

무드별 `id`, `mood_id`, `name_ko`, `name_en`, `description`, `keywords`, `color_palette`, `representatives` 등 포함.

### 9.3 실행

```bash
notebooks/04_build_library.ipynb
# 또는
python -c "from mood_pipeline.build_library import run_build_library; run_build_library()"
```

`run_build_library()` 완료 시 **Phase 6 검색 인덱스도 자동 생성**됩니다.

---

## 10. Phase 6 — 프롬프트 기반 무드 검색 (신규)

사용자가 무드를 직접 선택하지 않고, **자연어 프롬프트**로 가장 유사한 무드를 자동 추천합니다.

### 10.1 동작 흐름

```text
사용자 입력: "차분한 회색톤인데 너무 차갑지는 않은 원룸"
        ↓
CLIP text encoder (openai/clip-vit-base-patch32)
        ↓
[1차] mood_text_embeddings.npy 와 코사인 유사도 → Top-K 무드
        ↓
[2차] 1위 무드의 대표 이미지 embedding 과 비교 → Top-K 이미지
```

### 10.2 검색용 텍스트 구성

각 무드의 `index.json` + `meta.json`에서 아래 필드를 합쳐 `search_text` 생성:

- `name_ko`, `name_en`, `description`
- `keywords` / `tags`
- `typical_elements` (있을 경우)

### 10.3 저장 파일 (`mood_library/`)

| 파일 | shape / 형식 | 설명 |
|------|----------------|------|
| `mood_text_embeddings.npy` | (M, 512) | 무드 meta 텍스트 임베딩 |
| `mood_ids.json` | JSON array | mood_id, name_ko, search_text 등 |
| `mood_rep_image_embeddings.npy` | (R, 512) | cover + representatives 이미지 |
| `mood_rep_images.json` | JSON array | filename, path, mood_id, embedding_index |

### 10.4 API (`mood_pipeline/search.py`)

```python
from mood_pipeline.search import (
    build_mood_text_embeddings,   # 인덱스 빌드 (없으면 자동 생성)
    search_mood_by_prompt,        # 1차: 무드 Top-K
    search_images_within_mood,    # 2차: 특정 무드 내 이미지 Top-K
    search_mood_with_images,      # 1차 + 1위 무드 이미지 한번에
)

# 인덱스 빌드 (04_build_library 후 자동 생성됨, 수동 재빌드도 가능)
build_mood_text_embeddings()

# 1차 검색
results = search_mood_by_prompt("따뜻하고 우드톤이 있는 아늑한 원룸", top_k=3)
# [
#   {"mood_id": "warm_cozy", "mood_name_ko": "따뜻한 코지 원룸", "score": 0.80},
#   {"mood_id": "natural_wood", "mood_name_ko": "내추럴 우드", "score": 0.78},
#   ...
# ]

# 2차 검색
images = search_images_within_mood("따뜻한 우드톤", mood_id="natural_wood", top_k=5)

# 통합
full = search_mood_with_images("차분한 회색톤 원룸", top_k_moods=3, top_k_images=5)
# {"moods": [...], "selected_mood": {...}, "recommended_images": [...]}
```

### 10.5 실행

```bash
notebooks/05_search_mood.ipynb
```

### 10.6 참고

- 코사인 유사도 = L2 정규화 벡터의 **내적**
- 한국어 프롬프트도 동작하지만, meta에 영문(`name_en`)이 포함되어 있어 혼합 입력이 더 안정적일 수 있음
- 나중에 **강화학습 / preference learning**으로 프롬프트→무드 매칭 정확도 개선 가능 (현재 `mood_library`가 reward·레퍼런스 기준으로 사용)

---

## 11. 시각화·검증

| 도구 | 용도 |
|------|------|
| UMAP 2D scatter | 클러스터 분리 상태 |
| `data/cluster_previews/` montage | 잘못 묶인 이미지·라벨 확인 |
| Silhouette score | k 자동 선택 (현재 ~0.04대 → 원룸 데이터는 경계가 모호함) |
| 프롬프트 검색 결과 | `05_search_mood.ipynb`로 Top-K sanity check |

---

## 12. 추천 작업 순서 (체크리스트)

### 데이터 & 임베딩
- [ ] `images/final/` 300~500장 수집
- [ ] `01_clip_embed.ipynb` 또는 `run_pipeline.py`
- [ ] UMAP으로 분포 확인

### 클러스터링·라벨링
- [ ] `02_cluster.ipynb` — k 자동/수동
- [ ] `03_label.ipynb` — montage 검수, 필요 시 `labels.json` 수정

### 라이브러리·검색
- [ ] `04_build_library.ipynb` — `mood_library/` + 검색 인덱스
- [ ] `05_search_mood.ipynb` — 프롬프트로 무드 추천 테스트

---

## 13. 기술 스택 요약

```
크롤링:     Selenium (ohouse_crawl.ipynb, pinterest_board_crawl.ipynb)
임베딩:     transformers CLIP ViT-B/32 (이미지 + 텍스트)
클러스터링: scikit-learn KMeans
차원축소:   umap-learn (시각화)
분류:       CLIP zero-shot + 클러스터별 unique 라벨 배정
검색:       CLIP text embedding + 코사인 유사도 (2단계)
메타데이터: pandas, json
색상:       sklearn KMeans (cover 팔레트)
```

### requirements

```bash
pip install -r requirements-ml.txt
```

---

## 14. 구현 파일

| 파일 | 역할 |
|------|------|
| `mood_pipeline/preprocess.py` | 이미지 수집·검증·중복 제거 |
| `mood_pipeline/embed.py` | CLIP 이미지·텍스트 임베딩 |
| `mood_pipeline/cluster.py` | KMeans, UMAP, elbow |
| `mood_pipeline/label.py` | zero-shot 라벨, montage, metadata.csv |
| `mood_pipeline/build_library.py` | mood_library 빌드 + 검색 인덱스 |
| `mood_pipeline/search.py` | **프롬프트 기반 무드 검색** |
| `mood_pipeline/config.py` | 경로, MOOD_CANDIDATES, CLIP 모델 ID |
| `notebooks/01~05_*.ipynb` | 단계별 실행 |
| `run_pipeline.py` | 0~4단계 일괄 실행 (`--k`, `--no-dedup`, `--skip-viz`) |

### 일괄 실행

```bash
pip install -r requirements-ml.txt
python run_pipeline.py
# 검색만 다시 빌드:
python -c "from mood_pipeline.search import build_mood_text_embeddings; build_mood_text_embeddings(force=True)"
```

### 현재 설정

| 항목 | 값 |
|------|-----|
| `IMAGE_ROOT` | `images/final/` |
| CLIP 모델 | `openai/clip-vit-base-patch32` |
| 중복 제거 threshold | 0.95 |
| 최근 빌드 | 8무드, ~361장 (dedup 후) |

---

## 15. 흔한 문제 & 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| 모든 클러스터가 `warm_cozy` | CLIP zero-shot argmax 중복 | ✅ unique 라벨 배정 (`label.py`) |
| 클러스터 경계 모호 | 원룸 이미지 유사도 높음 | k 조정, 색감별 수집, montage 검수 |
| 프롬프트 검색이 엉뚱함 | meta 텍스트 빈약 | `typical_elements` 보강, 영문 description 추가 |
| `mood_text_embeddings.npy` 없음 | 04 미실행 | `build_mood_text_embeddings()` 또는 `04_build_library` 재실행 |
| 검색 인덱스 outdated | meta 수정 후 미갱신 | `build_mood_text_embeddings(force=True)` |

---

## 16. 한 줄 요약

> **CLIP으로 이미지 벡터화 → 클러스터링 → zero-shot 무드 라벨 → mood_library 패키징 → 사용자 프롬프트를 CLIP text embedding으로 변환해 가장 유사한 무드·대표 이미지를 자동 추천**

데이터 100장이면 k≈8, 500장이면 k≈12~15부터 실험하는 것을 권장합니다.

---

*작성 기준: `images/final/` 통합 데이터, CLIP ViT-B/32, mood_library 8무드 / 361장 (2026-06)*
