import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve
)
from scipy.stats import loguniform

url_sinasc = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-nascidos-vivos"
url_sisagua = "https://apidadosabertos.saude.gov.br/sisagua/controle-semestral"

params_url_sinasc = {"limit": 30000, "offset": 0}
params_url_sisagua = {"limit": 30000, "offset": 0, "semestre_de_referencia": 1}

#Constrói a pipeline para o classificador MLP binário.
def build_binary_pipeline():
    num_cols = ['ld', 'lq', 'resultado', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']
    cat_cols = ['parametro', 'grupo_de_parametros']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', MLPClassifier(
            tol=0.00001, n_iter_no_change=10,
            early_stopping=True, random_state=42
        ))
    ])
#Constrói a pipeline para o classificador MLP multiclasse.
def build_multiclass_pipeline():
    num_cols = ['ld', 'lq', 'resultado', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']
    cat_cols = ['parametro', 'grupo_de_parametros']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', MLPClassifier(
            tol=0.00001, n_iter_no_change=10,
            early_stopping=True, random_state=42
        ))
    ])
#Ajuste de hiperparametros
def tune_model(pipeline, param_dist, X_train, y_train, scoring='roc_auc', n_iter=50, fit_params=None):
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    if fit_params:
        random_search.fit(X_train, y_train, **fit_params)
    else:
        random_search.fit(X_train, y_train)

    print(f"Melhor score ({scoring}) no CV: {random_search.best_score_:.4f}")
    print(f"Melhores hiperparâmetros: {random_search.best_params_}")
    
    return random_search.best_estimator_

def evaluate_binary(model, X, y, prefix='binary'):
    y_proba = model.predict_proba(X)[:,1]
    p, r, t = precision_recall_curve(y, y_proba)
    f1 = 2 * p * r / (p + r + 1e-8)
    thresh = t[np.nanargmax(f1)] if len(t) > 0 else 0.5
    y_pred = (y_proba >= thresh).astype(int)
    
    print(f"\n[{prefix}] thresh={thresh:.3f}")
    print(classification_report(y, y_pred))
    print(f"[{prefix}] ROC AUC: {roc_auc_score(y, y_proba):.4f}")
    return y_pred


def main():
    response_sinasc = requests.get(url_sinasc, params=params_url_sinasc)
    response_sisagua = requests.get(url_sisagua, params=params_url_sisagua)

    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()["parametros"])
    
    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunnatu', right_on='codigo_ibge', how='left')
    
    merged_df['idanomal'] = pd.to_numeric(merged_df['idanomal'], errors='coerce').fillna(0).astype(int)
    merged_df = merged_df[merged_df['idanomal'].isin([1, 2])].copy()
    merged_df['tem_anomalia'] = (merged_df['idanomal'] == 1).astype(int)
    merged_df['codanomal'] = merged_df['codanomal'].astype(str).fillna('0')
    
    features_index = ['ld', 'lq', 'resultado', 'grupo_de_parametros', 'parametro', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']

    #Modelo binário
    X = merged_df[features_index]
    y = merged_df['tem_anomalia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    
    bin_pipe = build_binary_pipeline()
    
    param_dist_mlp_binary = {
        'model__max_iter': [1000, 1500, 2000],
        'model__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__solver': ['adam'],
        'model__alpha': loguniform(0.0001, 0.1),
        'model__learning_rate_init': loguniform(0.0001, 0.01),
        'model__batch_size': [32, 64, 128]
    }
    
    bin_model = tune_model(
        pipeline=bin_pipe,
        param_dist=param_dist_mlp_binary,
        X_train=X_train,
        y_train=y_train,
        scoring='roc_auc'
    )
    
    evaluate_binary(bin_model, X_train, y_train, prefix='MLP TREINO BINÁRIO')
    evaluate_binary(bin_model, X_test, y_test, prefix='MLP TESTE BINÁRIO')

    #Modelo multiclasse
    print("\n--- INICIANDO MODELAGEM MULTICLASSE (MLP) ---")
    df_multi = merged_df[merged_df['tem_anomalia'] == 1].copy()
    
    counts = df_multi['codanomal'].value_counts()
    rare = counts[counts < 10].index
    df_multi['codanomal_grouped'] = df_multi['codanomal'].where(~df_multi['codanomal'].isin(rare), 'Outros')
    
    X_multi = df_multi[features_index]
    y_multi = df_multi['codanomal_grouped']
    
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(
        X_multi, y_multi, test_size=0.25, stratify=y_multi, random_state=42
    )
    
    le = LabelEncoder()
    ym_train_enc = le.fit_transform(ym_train)
    
    multi_pipe = build_multiclass_pipeline()
    
    param_dist_mlp_multi = {
        'model__max_iter': [800, 1000, 1200],
        'model__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': loguniform(0.0001, 0.01),
        'model__learning_rate_init': loguniform(0.0001, 0.01),
        'model__batch_size': [64, 128]
    }
    
    multi_model = tune_model(
        pipeline=multi_pipe,
        param_dist=param_dist_mlp_multi,
        X_train=Xm_train,
        y_train=ym_train_enc,
        scoring='f1_macro'
    )
    
    ym_train_pred = le.inverse_transform(multi_model.predict(Xm_train))
    ym_test_pred = le.inverse_transform(multi_model.predict(Xm_test))

    print("\n--- Avaliação Final do Modelo MLP Multiclasse ---")
    print("[MLP multiclass - TREINO] Classification report:")
    print(classification_report(ym_train, ym_train_pred))
    print("\n[MLP multiclass - TESTE] Classification report:")
    print(classification_report(ym_test, ym_test_pred))

if __name__ == '__main__':
    main()