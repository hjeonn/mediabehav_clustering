# mediabehav_clustering
Unsupervised segmentation of media behavior data using Gower Distance and Agglomerative Clustering.

# 📊 Media Behavior Clustering Analysis

이 프로젝트는 미디어 이용 행위 데이터를 바탕으로 사용자 그룹을 세분화(Clustering)하는 머신러닝 분석 프로젝트입니다. Gower Distance와 계층적 군집 분석(Agglomerative Clustering)을 활용하여 복합 데이터를 분석합니다.

## 🚀 주요 기능
- **데이터 전처리**: `.sav` 파일 로드 및 범주형 데이터 인코딩
- **변수 선택**: `VarianceThreshold`를 이용한 저분산 피처 제거 및 데이터 정규화
- **거리 행렬 계산**: 범주형 변수를 고려한 `Gower Distance` 적용
- **최적 군집 탐색**: 실루엣 점수(Silhouette Score)를 이용한 최적의 K값 산출
- **시각화**: PCA(주성분 분석)를 통한 군집 결과 2차원 시각화

## 🛠️ 기술 스택
- **Language**: Python 3.x
- **Libraries**: 
  - `pandas`, `numpy` (데이터 처리)
  - `scikit-learn` (머신러닝 및 변수 선택)
  - `gower` (Gower 거리 계산)
  - `matplotlib` (시각화)
  - `pyreadstat` (SPSS 데이터 로드)

## 📂 프로젝트 구조
```text
.
├── main.py              # 분석 메인 스크립트
├── data/                # 데이터 파일 저장 폴더 (.sav 파일)
├── requirements.txt     # 필요 라이브러리 목록
└── README.md            # 프로젝트 설명 문서
