# Interior Mood Search

원룸 인테리어 이미지를 CLIP으로 분석해 **무드(mood) 라이브러리**를 만들고, **자연어 프롬프트**로 비슷한 무드·레퍼런스 이미지를 추천하는 프로젝트

```
프롬프트 → 무드 1개 선택 → gallery 이미지 추천
         ✅ 완료              ✅ 완료
```

---

## 환경 설정

```powershell
cd 창의학기

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ml.txt
```

- Python 3.10+ 
- 노트북 사용 시 커널을 **`.venv` Python**으로 선택

---

## 폴더 구조

```
창의학기/
├── images/final/          # 파이프라인 입력 이미지
├── data/                  # 임베딩·클러스터·메타데이터
├── mood_library/          # 무드별 cover, gallery, index.json
├── mood_pipeline/         # 핵심 Python 모듈
├── notebooks/             # 01~05 단계별 노트북
├── run_pipeline.py        # 전체 파이프라인 일괄 실행
└── search_mood.py         # 프롬프트 검색 CLI
```

---

## 사용 방법

### 1) 이미지 준비

`images/final/`에 원룸 인테리어 이미지를 보관

- 크롤러(선택): `ohouse_crawl.ipynb`(오늘의집), `pinterest_board_crawl.ipynb`(Pinterest)
- 수집 후 `images/final/`로 통합해 사용

### 2) 무드 라이브러리 만들기

방법 A — 노트북 (권장, 단계별 확인)

| 순서 | 노트북 | 내용 |
|------|--------|------|
| 1 | `notebooks/01_clip_embed.ipynb` | 전처리 + CLIP 임베딩 |
| 2 | `notebooks/02_cluster.ipynb` | KMeans 클러스터링 |
| 3 | `notebooks/03_label.ipynb` | zero-shot 무드 라벨링 |
| 4 | `notebooks/04_build_library.ipynb` | `mood_library/` 생성 |

**방법 B — 한 번에 실행**

```powershell
python run_pipeline.py
```

옵션:

```powershell
python run_pipeline.py --k 8          # 클러스터 수 직접 지정
python run_pipeline.py --no-dedup     # 중복 제거 끄기
python run_pipeline.py --skip-viz     # UMAP/elbow 생략
```

### 3) 프롬프트로 이미지 검색

**방법 A — 노트북**

`notebooks/05_search_mood.ipynb` → **셀 2 → 셀 3** 순서 실행

```python
prompt = "세련되고 고급스러운 방"
result = search_mood_with_images(prompt, top_k=5)  # gallery 최대 5장
plot_search_result(result, prompt=result.get("prompt_en", prompt))
```

- 한국어 입력 시 **자동 영어 번역** 후 CLIP 검색
- `top_k`: 추천 gallery 이미지 개수 (무드 cover는 항상 1장)

**방법 B — 터미널**

```powershell
python search_mood.py "세련되고 고급스러운 방" --top-k 5 --plot
```

옵션:

| 옵션 | 설명 |
|------|------|
| `--top-k 5` | gallery 추천 개수 |
| `--plot` | matplotlib figure 표시 |
| `--no-translate` | 한→영 번역 끄기 |
| `--moods-only` | 무드 검색만 (이미지 추천 생략) |

---

## 검색 동작 요약

1. 사용자 프롬pt → CLIP 텍스트 임베딩 (한국어는 영어로 번역)
2. **1차** — 8개 무드 meta와 유사도 비교 → **1위 무드** 선택
3. **2차** — 1위 무드 gallery 전체에서 프롬프트와 가장 비슷한 이미지 `top_k`장 추천
4. score = 코사인 유사도 (-1~1, **1에 가까울수록 유사**, 텍스트-이미지는 0.2~0.3대도 정상)

---

## 현재 생성된 무드 (8개)

| mood_id | 이름 |
|---------|------|
| `natural_wood` | 내추럴 우드 |
| `minimal_white` | 미니멀 화이트 |
| `bright_airy` | 밝은 에어리 |
| `cute_pastel` | 파스텔 귀여운 |
| `monochrome_minimal` | 모노톤 미니멀 |
| `vintage_retro` | 빈티지 레트로 |
| `warm_cozy` | 따뜻한 코지 원룸 |
| `luxury_modern` | 럭셔리 모던 |

---

## 참고 문서

| 파일 | 내용 |
|------|------|
| `MOOD_LIBRARY_PIPELINE.md` | 파이프라인 상세 가이드 |
| `result.md` | 완료 현황·산출물 정리 |
