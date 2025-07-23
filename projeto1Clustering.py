import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import umap.umap_ as umap
from kneed import KneeLocator
import numpy as np

url_sinasc = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-nascidos-vivos"
url_sisagua = "https://apidadosabertos.saude.gov.br/sisagua/controle-semestral"

params_url_sinasc = {
    "limit": 30000,
    "offset": 0
}

params_url_sisagua = {
    "limit": 30000,
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

def plot_pca_variance_cumsum(pca, plot_dir = None):
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

def k_optimizer(data, max_k=20, plot_dir=None):
    inertias = []
    k_range = range(1, max_k)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inércia', fontsize=12)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label='Vertical Line')
    plt.title('Método do Cotovelo para K-means (nos componentes PCA)', fontsize=14)

    plt.grid(True)
    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(plot_dir + 'inertia.png')
    else:
        plt.savefig('plots/' + 'inertia.png')

    return optimal_k

def visualize_clustering(values, labels, plot_dir=None):

    tsne = TSNE(random_state=42, verbose=1, max_iter=2000, n_components=2)
    tsne_data = tsne.fit_transform(values)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(values)

    umap_data = umap.UMAP(random_state=42).fit_transform(values)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(x=pca_data[:, 0], y=pca_data[:, 1], c=labels, cmap='viridis')
    plt.title("PCA Reduction")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.colorbar()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.scatter(x=tsne_data[:, 0], y=tsne_data[:, 1], c=labels, cmap='viridis')
    plt.title("TSNE Reduction")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.colorbar()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.scatter(x=umap_data[:, 0], y=umap_data[:, 1], c=labels, cmap='viridis')
    plt.title("UMAP Reduction")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.colorbar()

    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(plot_dir + 'cluster_analysys.png')
    else:
        plt.savefig('plots/' + 'cluster_analysys.png')

def main():

    plot_dir = 'plots'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    print(">>>Coletado dados da API<<<")

    response_sinasc = requests.get(url_sinasc, params=params_url_sinasc)
    response_sisagua = requests.get(url_sisagua, params=params_url_sisagua)
    
    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()["parametros"])

    print(">>>Realizando merge dos dois dataframes<<<")

    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunres', right_on='codigo_ibge', how='left')

    print(f">>>Imprimindo gráficos referentes aos dados originais em {plot_dir}/<<<")

    plot_oritinal_distributions(merged_df)
   
    features_index = ['ld', 'lq', 'resultado', 'grupo_de_parametros','parametro']
    #target_index = ['codanomal', 'idanomal']
    target_index = ['codanomal']
    #target_index = ['idanomal']

    categorical_cols = ['grupo_de_parametros', 'parametro'] + target_index
    
    print(f">>>Realizando pré-processamento dos dados com as colunas: {features_index + target_index}<<<")

    preprocessed_data = preprocessor_func(merged_df, features_index + target_index, categorical_columns=categorical_cols)

    pca = PCA()
    pca.fit_transform(preprocessed_data)

    plot_pca_variance_cumsum(pca)

    pca = PCA(n_components=0.9)
    dim_reduced_data = pca.fit_transform(preprocessed_data)

    print(">>>Calculando k ótimo<<<")

    optimal_k_pca = k_optimizer(dim_reduced_data, max_k=200)

    print(">>>Realizando agrupamento via Kmeans<<<")

    kmeans = KMeans(n_clusters=optimal_k_pca, random_state=42)
    merged_df['cluster_kmeans'] = kmeans.fit_predict(dim_reduced_data)

    print(">>>Realizando agrupamento via DBSCAN<<<")

    dbscan = DBSCAN(eps=5, min_samples=39)
    merged_df['cluster_dbscan'] = dbscan.fit_predict(dim_reduced_data)

    print(">>>Gerando resultados do agrupamento<<<")

    visualize_clustering(preprocessed_data, merged_df['cluster_kmeans'])

    print(f">>>Resultados do agrupamento impressos em {plot_dir}/<<<")

    
if __name__ == "__main__":
    main()