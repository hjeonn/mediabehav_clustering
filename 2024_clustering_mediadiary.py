import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gower
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# 1. 데이터 로드 설정
# 로컬 환경이나 GitHub 레포지토리 구조에 맞게 경로를 수정하여 사용하세요.
FILE_PATH = 'data/20년미디어행위9-24.sav'

def load_data(path):
    print(f"Loading data from: {path}")
    df = pd.read_spss(path)
    return df

# 2. 전처리 함수
def preprocess_data(df):
    # 클러스터링에 사용할 feature 추출 (8번째 컬럼부터)
    processed_df = df.iloc[:, 7:].copy()
    
    # Categorical 데이터를 Numerical로 변환 (Factorize)
    for col in processed_df.columns:
        if processed_df[col].dtype == 'category':
            processed_df[col] = pd.factorize(processed_df[col])[0]
            
    # 결측치 처리 (필요 시 0으로 채움)
    processed_df = processed_df.fillna(0).astype(float)
    return processed_df

# 3. 변수 선택 (Variance Threshold)
def select_features(df, threshold=0.043):
    # 정규화 진행 후 변수 선택
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    
    selector = VarianceThreshold(threshold=threshold)
    selected_data = selector.fit_transform(normalized_data)
    
    selected_indices = selector.get_support(indices=True)
    selected_names = df.columns[selected_indices].tolist()
    
    print(f"Original features: {df.shape[1]}")
    print(f"Selected features: {len(selected_names)}")
    
    return pd.DataFrame(selected_data, columns=selected_names)

# 4. 시각화 함수 (PCA 기반)
def visualize_clusters(data, labels, title="Cluster Visualization"):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        cluster_points = pca_result[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {label}')
    
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.show()

# --- 메인 실행 흐름 ---

# 데이터 로드 및 전처리
raw_data = load_data(FILE_PATH)
processed_data = preprocess_data(raw_data)
selected_df = select_features(processed_data)

# Gower Distance 계산
print("Calculating Gower distance matrix...")
dist_matrix = gower.gower_matrix(processed_data)

# 최적의 K 찾기 (실루엣 분석)
range_n_clusters = range(2, 7)
silhouette_avg_scores = []

for n in range_n_clusters:
    clusterer = AgglomerativeClustering(n_clusters=n, linkage='complete')
    labels = clusterer.fit_predict(dist_matrix)
    score = silhouette_score(dist_matrix, labels, metric='precomputed')
    silhouette_avg_scores.append(score)
    print(f"For n_clusters = {n}, Silhouette Score: {score:.4f}")

# 결과 시각화 (K=3 예시)
optimal_k = 3
final_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='complete')
final_labels = final_model.fit_predict(dist_matrix)

visualize_clusters(processed_data, final_labels, title=f"Agglomerative Clustering (k={optimal_k})")

# 결과 저장
raw_data['cluster_label'] = final_labels
# raw_data.to_csv('clustering_results.csv', index=False)
print("Analysis complete.")