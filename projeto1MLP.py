import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import LabelEncoder


url_sinasc = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-nascidos-vivos"
url_sisagua = "https://apidadosabertos.saude.gov.br/sisagua/controle-semestral"

le = LabelEncoder()

params_url_sinasc = {
    "limit": 30000,
    "offset": 0
}

params_url_sisagua = {
    "limit": 30000,
    "offset": 0,
    "semestre_de_referencia": 1
}




def build_binary_pipeline():
    num_cols = ['ld','lq','resultado','idademae','gravidez','gestacao','consprenat','mesprenat']
    cat_cols = ['parametro','grupo_de_parametros']
    numeric = Pipeline([('imp', SimpleImputer(strategy='median')),('scale', StandardScaler())])
    categorical = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                            ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', numeric, num_cols),('cat', categorical, cat_cols)])

    mlp = MLPClassifier(max_iter=3000, tol=1e-5, n_iter_no_change=10,
                        early_stopping=True, random_state=42)
    pipe = ImbPipeline([('pre', pre),('smote', SMOTE(random_state=42, k_neighbors=3)),('mlp', mlp)])
    return pipe


def ajuste_binary(pipe, X, y):
    dist = {
        'smote__sampling_strategy': [0.5,0.6,0.7,0.8,1.0],
        'mlp__hidden_layer_sizes': [(50,),(100,),(50,25),(100,50)],
        'mlp__activation': ['relu','tanh'],
        'mlp__solver': ['adam','sgd'],
        'mlp__alpha': loguniform(1e-4,1e-1),
        'mlp__learning_rate_init': loguniform(1e-4,1e-2),
        'mlp__batch_size': [32,64,128]
    }
    search = HalvingRandomSearchCV(pipe, dist, resource='mlp__max_iter',
                                  max_resources=3000, min_resources=300,
                                  factor=2, cv=5, scoring='recall',
                                  random_state=42, n_jobs=-1, verbose=1)
    search.fit(X, y)
    best_pipe = search.best_estimator_
    print(f"Melhores parametros - BINARIO: {search.best_params_}")
    calib = CalibratedClassifierCV(best_pipe, cv=5, method='isotonic')
    calib.fit(X, y)
    return calib, search.best_params_


def evaluate_binary(model, X, y, prefix='binary'):

    y_proba = model.predict_proba(X)[:,1]
    p,r,t = precision_recall_curve(y, y_proba)
    f1 = 2*p*r/(p+r+1e-8)
    thresh = t[np.nanargmax(f1)] if len(t)>0 else 0.5
    y_pred = (y_proba >= thresh).astype(int)
    print(f"[{prefix}] thresh={thresh:.3f}")
    print(classification_report(y, y_pred))
    print('ROC AUC:', roc_auc_score(y, y_proba))
    return y_pred


def build_multiclass_pipeline(best_params):
    num_cols = ['ld','lq','resultado','idademae','gravidez','gestacao','consprenat','mesprenat']
    cat_cols = ['parametro','grupo_de_parametros']
    numeric = Pipeline([('imp', SimpleImputer(strategy='median')),('scale', StandardScaler())])
    categorical = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                            ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', numeric, num_cols),('cat', categorical, cat_cols)])

    mlp = MLPClassifier(
        hidden_layer_sizes=best_params['mlp__hidden_layer_sizes'],
        activation=best_params['mlp__activation'],
        solver=best_params['mlp__solver'],
        alpha=best_params['mlp__alpha'],
        learning_rate_init=best_params['mlp__learning_rate_init'],
        batch_size=best_params['mlp__batch_size'],
        learning_rate=best_params.get('mlp__learning_rate','adaptive'),
        max_iter=3000, tol=1e-5, n_iter_no_change=10,
        early_stopping=True, random_state=42
    )
    pipe = ImbPipeline([('pre', pre),('smote', SMOTE(random_state=42)),('mlp', mlp)])
    return pipe


def main():

    response_sinasc = requests.get(url_sinasc, params=params_url_sinasc)
    response_sisagua = requests.get(url_sisagua, params=params_url_sisagua)

    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()["parametros"])

    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunres', right_on='codigo_ibge', how='left')
    merged_df['idanomal'] = pd.to_numeric(merged_df['idanomal'], errors='coerce').fillna(0).astype(int)
    #Filtra os casos em que não foi registrada se havia ou não anomalia
    merged_df = merged_df[merged_df['idanomal'].isin([1, 2])].copy()
    merged_df['tem_anomalia'] = (merged_df['idanomal'] == 1).astype(int)
    merged_df['codanomal'] = merged_df['codanomal'].astype(str).fillna('0')

    features_index = ['ld', 'lq', 'resultado', 'grupo_de_parametros','parametro', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']


    # Primeiro verificar se tem anomalia ou não
    X = merged_df[features_index]
    y = merged_df['tem_anomalia']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    bin_pipe = build_binary_pipeline()
    bin_model, best_params = ajuste_binary(bin_pipe, X_train, y_train)
    evaluate_binary(bin_model, X_train, y_train, prefix='treino binario')
    evaluate_binary(bin_model, X_test, y_test, prefix='teste binario')

    df_multi = merged_df[merged_df['tem_anomalia'] == 1].copy()
    counts = df_multi['codanomal'].value_counts()
    rare = counts[counts < 10].index
    df_multi['codanomal_grouped'] = df_multi['codanomal'].where(~df_multi['codanomal'].isin(rare), 'Outros')
    X_multi = df_multi[features_index]
    y_multi = df_multi['codanomal_grouped']
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(
        X_multi, y_multi, test_size=0.25, stratify=y_multi, random_state=42
    )
    ym_train_enc = le.fit_transform(ym_train)
    multi_pipe = build_multiclass_pipeline(best_params)
    multi_pipe.fit(Xm_train, ym_train_enc)
    y_pred_enc = multi_pipe.predict(Xm_test)     
    y_pred_multi = le.inverse_transform(y_pred_enc)
    print("[multiclass] Classification report:")
    print(classification_report(ym_test, y_pred_multi))

if __name__ == '__main__':
    main()
