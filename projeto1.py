import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

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

    print(f"Colunas importadas do dataset Sinasc:\n{df_sinasc.columns}")
    print(f"Colunas importadas do dataset Sisagua:\n{df_sisagua.columns}")

    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunnatu', right_on='codigo_ibge', how='left')

    print(f"Dados agregados de ambos os datasets:\n{merged_df.columns}")

    features_index = ['ld', 'lq', 'resultado', 'parametro', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat']
    target_index = ['codanomal', 'idanomal', 'tpnascassi']

    X = merged_df[features_index]
    y = merged_df[target_index]

    contagem_anomalias = y['idanomal'].value_counts()
    labels = [f"{idx} (n={v})" for idx, v in contagem_anomalias.items()]
    
    plt.figure(figsize=(10, 8))
    

    plt.pie(contagem_anomalias, labels=labels)
    plt.title('Distribuição de anomalias congênitas')
    plt.xlabel('Tipo de anomalia')
    plt.ylabel('Frequência')
    plt.savefig('plots/' + 'dist_anomal.png')



if __name__ == "__main__":
    main()