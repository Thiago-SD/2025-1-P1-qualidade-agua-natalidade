import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np

url_sinasc = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-nascidos-vivos"
url_sisagua = "https://apidadosabertos.saude.gov.br/sisagua/controle-semestral"

params_url_sinasc = {
    "limit": 10000,
    "offset": 0
}

params_url_sisagua = {
    "limit": 10000,
    "offset": 0,
    "semestre_de_referencia": 1
}

def main():

    plot_folder = 'plots'
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    response_sinasc = requests.get(url_sinasc, params=params_url_sinasc)
    response_sisagua = requests.get(url_sisagua, params=params_url_sisagua)
    
    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()["parametros"])

    #print(f"Colunas importadas do dataset Sinasc:\n{df_sinasc.columns}")
    #print(f"Colunas importadas do dataset Sisagua:\n{df_sisagua.columns}")

    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunnatu', right_on='codigo_ibge', how='left')

    #print(f"Dados agregados de ambos os datasets:\n{merged_df.columns}")

    contagem_contaminantes = merged_df['grupo_de_parametros'].value_counts()
    labels = [f"{idx} (n={v})" for idx, v in contagem_contaminantes.items()]

    plt.figure(figsize=(6, 4))
    plt.pie(contagem_contaminantes, labels=labels)
    plt.title('Distribuição de Parametros Observados')
    plt.savefig('plots/' + 'dist_param.png')

    contagem_anomalias = merged_df['idanomal'].value_counts()
    labels = [f"{idx} (n={v})" for idx, v in contagem_anomalias.items()]
    
    plt.figure(figsize=(6, 4))

    plt.pie(contagem_anomalias, labels=labels)
    plt.title('Distribuição de anomalias congênitas')
    plt.xlabel('Tipo de anomalia')
    plt.ylabel('Frequência')
    plt.savefig('plots/' + 'dist_anomal.png')

    contagem_anomalias = pd.DataFrame(merged_df['codanomal'].value_counts())

    plt.figure(figsize=(10, 6))
    sns.barplot(data=contagem_anomalias, x='codanomal', y='count')


    plt.title('Contagem de Anomalias por Código e Tipo de Nascimento')
    plt.xlabel('Código da Anomalia')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/' + 'cont_anomal.png')

    features_index = ['ld', 'lq', 'resultado', 'grupo_de_parametros','parametro', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']
    target_index = ['codanomal', 'idanomal']

    categorical_cols = ['grupo_de_parametros', 'parametro', 'gravidez', 'gestacao', 'tpnascassi', 'codanomal', 'idanomal']
    cluster_data = merged_df[features_index + target_index].copy()
    cluster_data = pd.get_dummies(cluster_data, columns=categorical_cols)

    imputer = SimpleImputer(strategy='median')
    cluster_data_imputed = imputer.fit_transform(cluster_data)

    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data_imputed)

    pca = PCA()
    pca.fit_transform(cluster_data_scaled)

    plt.figure(figsize=(8, 6))
    plt.plot([i for i in range(1, pca.n_components_ + 1)], np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Número de Componentes Principais')
    plt.ylabel('Fração Cumulativa da Variância Explicada')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% da variância')
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% da variância')
    plt.title('Gráfico de Fração Cumulativa da Variância Explicada')
    plt.grid(True)
    plt.savefig('plots/' + 'cumsum_func.png')

    pca = PCA(n_components=0.9)
    cluster_data_reduced = pca.fit_transform(cluster_data_scaled)

    inertias = []
    k_range = range(1, len(contagem_anomalias) + 140)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cluster_data_reduced)
        inertias.append(kmeans.inertia_)

    # Plot do método do cotovelo
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inércia', fontsize=12)
    plt.title('Método do Cotovelo para K-means (nos componentes PCA)', fontsize=14)
    plt.savefig('plots/' + 'inertia.png')
    plt.grid(True)

    optimal_k = 125
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters_kmeans = kmeans.fit_predict(cluster_data_reduced)

    dbscan = DBSCAN(eps=10, min_samples=41)
    clusters_dbscan = dbscan.fit_predict(cluster_data_reduced)

    merged_df['cluster_kmeans'] = clusters_kmeans
    merged_df['cluster_dbscan'] = clusters_dbscan

    nascimentos_anomalia = merged_df.dropna(subset=['codanomal'])

    contagem_clusters = pd.DataFrame(nascimentos_anomalia['cluster_kmeans'].value_counts())
    contagem_clusters.head()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=contagem_clusters, x='cluster_kmeans', y='count')


    plt.title('Resultados do Agrupamento(Kmeans)')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.savefig('plots/' + 'results_kmeans.png')

    nascimentos_anomalia = merged_df.dropna(subset=['codanomal'])

    contagem_clusters = pd.DataFrame(nascimentos_anomalia['cluster_dbscan'].value_counts())
    contagem_clusters.head()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=contagem_clusters, x='cluster_dbscan', y='count')


    plt.title('Resultados do Agrupamento(DBSCAN)')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.savefig('plots/' + 'results_dbscan.png')
    
if __name__ == "__main__":
    main()