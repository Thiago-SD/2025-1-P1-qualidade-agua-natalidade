import os
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn

def inner_merge(left_array, right_array, left_key, right_key):
    left_keys = left_array[left_key]
    right_keys = right_array[right_key]
    
    common_keys, left_indices, right_indices = np.intersect1d(
        left_keys, right_keys, 
        assume_unique=False, 
        return_indices=True
    )
    
    left_filtered = left_array[left_indices]
    right_filtered = right_array[right_indices]
    
    right_names = [name for name in right_array.dtype.names if name != right_key]
    new_dtype = left_array.dtype.descr + [d for d in right_array.dtype.descr if d[0] in right_names]
    
    merged = np.empty(len(left_filtered), dtype=new_dtype)
    
    for name in left_array.dtype.names:
        merged[name] = left_filtered[name]
    
    for name in right_names:
        merged[name] = right_filtered[name]
    
    return merged

# 1- Carregando os dados iniciais

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'data')

# Carrega apenas as colunas necessárias para economizar memória
cols_sinasc = ['id_municipio_nascimento', 'id_anomalia', 'codigo_anomalia', 'peso', 'apgar1', 'apgar5']
cols_snis = [
    'id_municipio', 
    'populacao_urbana', 
    'investimento_total_municipio', 
    'populacao_urbana_atendida_agua_ibge', 
    'populacao_urbana_residente_esgoto_ibge',
    'indice_coleta_esgoto',
    'indice_tratamento_esgoto',
    'indice_perda_distribuicao_agua'
]

# Lê os CSVs diretamente para arrays NumPy (usando pandas apenas para leitura inicial)
raw_sinasc = pd.read_csv(os.path.join(data_path, "br_mdr_sinasc_nascidos_vivos_2022.csv"), 
                        usecols=cols_sinasc, 
                        low_memory=False)
raw_snis = pd.read_csv(os.path.join(data_path, "br_mdr_snis_municipio_agua_esgoto.csv"), 
                      usecols=cols_snis)

# Converte para arrays NumPy estruturados
sinasc_array = raw_sinasc.to_records(index=False)
snis_array = raw_snis.to_records(index=False)

print(f"Estatísticas dos datasets brutos:\n-->Nascidos Vivos:\n{sinasc_array[:5]}\n-->Saneamento Básico:\n{snis_array[:5]}")

# 2 - Tratando os dados para análise

merged_data = inner_merge(sinasc_array, snis_array, 'id_municipio_nascimento', 'id_municipio')

print(f"Estatísticas do dataset agregado:\n-->Dataset Agregado:\n{merged_data[:5]}")
#full_dataset = pd.merge(raw_sinasc_dataset[cols_sinasc], raw_snis_dataset [cols_snis], left_on='id_municipio_nascimento', right_on='id_municipio', how='inner')

#print(f"Estatísticas do dataset completo pós merge:\n-->Dataset Completo:\n{full_dataset.describe()}")