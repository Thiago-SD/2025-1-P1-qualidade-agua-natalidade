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

def plot_oritinal_distributions(data, plot_dir=None):
    contagem_contaminantes = data['grupo_de_parametros'].value_counts()
    labels = [f"{idx} (n={v})" for idx, v in contagem_contaminantes.items()]

    plt.figure(figsize=(16, 18))
    
    plt.subplot(2, 2, 1)
    plt.pie(contagem_contaminantes, autopct='%1.1f%%')
    plt.legend(labels, loc='lower left', fontsize='x-small')
    plt.title('Distribuição de Parametros Observados')
    plt.grid(True)

    contagem_nascimentos = data['idanomal'].value_counts()
    labels = [f"{idx} (n={v})" for idx, v in contagem_nascimentos.items()]

    plt.subplot(2, 2, 2)
    plt.pie(contagem_nascimentos, autopct='%1.1f%%')
    plt.title('Distribuição de anomalias congênitas Identificadas')
    plt.legend(labels, loc='lower left', title='Anomalia identificada', fontsize='x-small')
    plt.grid(True)

    contagem_anomalias = pd.DataFrame(data['codanomal'].value_counts())


    plt.subplot(2, 1, 2)
    sns.barplot(data=contagem_anomalias, x='codanomal', y='count', width=0.9)
    plt.title('Contagem de Anomalias por Código')
    plt.xlabel('Código da Anomalia')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    if plot_dir is not None:
        plt.savefig(plot_dir + 'stats_original_data.png')
    else:
        plt.savefig('plots/' + 'stats_original_data.png')

def preprocessor_func(data, indexes=None, categorical_columns=None):
    
    if indexes is not None:
        cluster_data = data[indexes].copy()
    else:
        cluster_data = data.copy()

    if categorical_columns is not None:
        cluster_data = pd.get_dummies(cluster_data, columns=categorical_columns)

    imputer = SimpleImputer(strategy='median')
    cluster_data_imputed = imputer.fit_transform(cluster_data)

    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data_imputed)

    return cluster_data_scaled

def dim_reduction(data, plot_dir = None):
    pca = PCA()
    pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.plot([i for i in range(1, pca.n_components_ + 1)], np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Número de Componentes Principais')
    plt.ylabel('Fração Cumulativa da Variância Explicada')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% da variância')
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% da variância')
    plt.title('Gráfico de Fração Cumulativa da Variância Explicada')
    plt.grid(True)
    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(plot_dir + 'cumsum_func.png')
    else:
        plt.savefig('plots/' + 'cumsum_func.png')

    pca = PCA(n_components=0.9)
    return pca.fit_transform(data)

def k_optimizer(data, max_k=20, plot_dir=None):

    inertias = []
    k_range = range(1, max_k)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    deltas = np.diff(inertias)
    deltas2 = np.diff(deltas)
    acceleration = np.concatenate(([0], deltas2))
    
    optimal_k = np.argmin(acceleration) + 1

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inércia', fontsize=12)
    plt.title('Método do Cotovelo para K-means (nos componentes PCA)', fontsize=14)

    plt.grid(True)
    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(plot_dir + 'inertia.png')
    else:
        plt.savefig('plots/' + 'inertia.png')

    return optimal_k

def plot_clustering_results(data, plot_dir=None):

    cluster_counts_kmeans = pd.DataFrame(data['cluster_kmeans'].value_counts())
    cluster_counts_dbscan = pd.DataFrame(data['cluster_dbscan'].value_counts())

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    sns.barplot(data=cluster_counts_kmeans, x='cluster_kmeans', y='count')
    plt.title('Resultados do Agrupamento(Kmeans)')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    sns.barplot(data=cluster_counts_dbscan, x='cluster_dbscan', y='count')
    plt.title('Resultados do Agrupamento(DBSCAN)')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)


    if plot_dir is not None:
        plt.savefig(plot_dir + 'results_clustering.png')
    else:
        plt.savefig('plots/' + 'results_clustering.png')

def main():

    plot_folder = 'plots'
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    response_sinasc = requests.get(url_sinasc, params=params_url_sinasc)
    response_sisagua = requests.get(url_sisagua, params=params_url_sisagua)
    
    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()["parametros"])

    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunnatu', right_on='codigo_ibge', how='left')

    plot_oritinal_distributions(merged_df)
   
    features_index = ['ld', 'lq', 'resultado', 'grupo_de_parametros','parametro', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']
    target_index = ['codanomal', 'idanomal']

    categorical_cols = ['grupo_de_parametros', 'parametro', 'gravidez', 'gestacao', 'tpnascassi', 'codanomal', 'idanomal']
    
    preprocessed_data = preprocessor_func(merged_df, features_index + target_index, categorical_columns=categorical_cols)

    reduced_data = dim_reduction (preprocessed_data)

    k_range = range(1, len(reduced_data['codanomal'].value_counts()))

    optimal_k = k_optimizer (reduced_data, max_k=k_range)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters_kmeans = kmeans.fit_predict(reduced_data)

    dbscan = DBSCAN(eps=10, min_samples=41)
    clusters_dbscan = dbscan.fit_predict(reduced_data)

    merged_df['cluster_kmeans'] = clusters_kmeans
    merged_df['cluster_dbscan'] = clusters_dbscan

    nascimentos_anomalia = merged_df.dropna(subset=['codanomal'])

    plot_clustering_results(nascimentos_anomalia)

    
if __name__ == "__main__":
    main()