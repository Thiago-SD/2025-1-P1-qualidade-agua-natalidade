import requests
import pandas as pd

# URL da API para o Sistema de Informação sobre Nascidos Vivos
url_sinasc = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-nascidos-vivos"
url_sisagua = "https://apidadosabertos.saude.gov.br/sisagua/controle-semestral"

# Parâmetros da requisição (ajuste conforme necessário)
params = {
    "limit": 100,  # Número máximo de registros por página
    "offset": 0     # Deslocamento para paginação
}

response_sinasc = requests.get(url_sinasc, params=params)
response_sisagua = requests.get(url_sisagua, params=params)
 
df_sinasc = pd.DataFrame(response_sinasc.json())
df_sisagua = pd.DataFrame(response_sisagua.json())

print(df_sisagua.describe())