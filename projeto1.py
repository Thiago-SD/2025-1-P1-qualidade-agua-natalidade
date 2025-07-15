import requests
import pandas as pd

url_sinasc = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-nascidos-vivos"
url_sisagua = "https://apidadosabertos.saude.gov.br/sisagua/controle-semestral"

params = {
    "limit": 100,  
    "offset": 0     
}

def main():
    response_sinasc = requests.get(url_sinasc, params=params)
    response_sisagua = requests.get(url_sisagua, params=params)
    
    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()['parametros'])

    print(f"Colunas importadas do dataset Sinasc:\n{df_sinasc.head()}")
    print(f"Colunas importadas do dataset Sisagua:\n{df_sisagua.head()}")

    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunnatu', right_on='codigo_ibge', how='left')

    print(f"Dados agregados de ambos os datasets:\n{merged_df.head()}")

if __name__ == "__main__":
    main()