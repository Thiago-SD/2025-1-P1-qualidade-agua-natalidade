import os
import pandas as pd

# 1- Carregando os dados iniciais

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'data')

raw_sinasc_dataset = pd.read_csv(os.path.join(data_path, "br_mdr_sinasc_nascidos_vivos_2022.csv"), low_memory=False)
raw_snis_dataset = pd.read_csv(os.path.join(data_path, "br_mdr_snis_municipio_agua_esgoto.csv"))

print(f"Estatísticas dos datasets brutos:\n-->Nascidos Vivos:\n{raw_sinasc_dataset.describe()}\n-->Saneamento Básico:\n{raw_snis_dataset.describe()}")

# 2 - Tratando os dados para análise

features = raw_snis_dataset['populacao_urbana', 'investimento_total_municipio', 'populacao_urbana_atendida_agua_ibge', 'populacao_urbana_residente_esgoto_ibge']

target = raw_sinasc_dataset['id_anomalia', 'codigo_anomalia']

#full_dataset = pd.merge(raw_sinasc_dataset[cols_sinasc], raw_snis_dataset [cols_snis], left_on='id_municipio_nascimento', right_on='id_municipio', how='inner')

#print(f"Estatísticas do dataset completo pós merge:\n-->Dataset Completo:\n{full_dataset.describe()}")