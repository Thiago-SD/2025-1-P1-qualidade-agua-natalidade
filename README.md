# 2025-1-P1-qualidade-agua-natalidade

## Primeiro Passo - Configurando o ambiente

Para garantir que os scripts sejam executados em um ambiente contendo todas as dependências necessárias, deve ser criado um ambiente virtual 

```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```

Após a criação e ativação do ambiente virtual, instalar as dependências necessárias:

```
pip install -r requirements.txt
```

## Segundo Passo - Execução dos scripts

Os scripts python, nomeados de acordo com sua função, podem ser executados uma vez que o ambiente virtual está configurado, podem ser executados com a versão atual do python instalada

Os dados de entrada possuem parâmetros que podem ser alterados, especicamente os campos "limit" e "offset" que correspondem, respectivamente, ao limite de entradas por requisição da API e número de entradas ignoradas por uma requisição, fazendo um "shift down" dos dados na API 

<img width="341" height="205" alt="image" src="https://github.com/user-attachments/assets/8baf3d45-c319-4cdb-a692-638b788efb14" />

Os passos de pré-processamento são executados antes de cada análise, é também impressa uma visão geral dos dados de entrada (obs: o diretório padrão onde os plots são salvos é ./plot):

<img width="1600" height="1800" alt="image" src="https://github.com/user-attachments/assets/05f5a802-e35d-4b32-8792-481598ebd8ff" />


Certos campos podem ser alterados conforme necessidade, como por exemplo, o valor de k, contendo os clusters do algoritmo k-means

```
    optimal_k_pca = k_optimizer(dim_reduced_data, max_k=200)
```

O gráfico da função k_optimizer é plotado caso o valor precise ser alterado manualmente 

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/d12eefa3-9cd9-4872-82b4-8b07a64ddf38" />

Algo semelhante pode ser feito com o PCA, decidindo quantos componentes são necessários para análise, a função de variância cumulativa também é plotada e pode ser consultada:

```
    pca = PCA(n_components=0.9)
```

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/e39d7a09-b5b9-4230-9612-02a7080be75b" />

Os scripts de treino de classificadores já fazem uma seleção de hiperparâmetros, porém novos valores podem ser inseridos nos dicionários param_dist caso seja desejado, conforme exemplo:

```
param_dist = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [3, 5, 7, 9],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],         
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],  
        'model__gamma': [0, 0.1, 0.5, 1],               
        'model__min_child_weight': [1, 3, 5]            
    }
```

Os resultados do processo de classificação são impressos no terminal de execução após o treino e teste, conforme exemplo:

<img width="1357" height="503" alt="image" src="https://github.com/user-attachments/assets/562a71d1-c98b-4ce7-b9ef-afc8f18ba4ae" />

## Link para o Vídeo:

https://www.youtube.com/watch?v=L_6gSRgAJeU



