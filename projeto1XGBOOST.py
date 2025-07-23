import numpy as np
import pandas as pd
import requests
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import compute_sample_weight
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

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

def build_and_tune_xgboost(X_train, y_train):
    """
    Constrói uma pipeline de pré-processamento e um modelo XGBoost,
    e ajusta os hiperparâmetros usando RandomizedSearchCV.
    """
    num_cols = ['ld', 'lq', 'resultado', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']
    cat_cols = ['parametro', 'grupo_de_parametros']
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    #Escala para balanceamento
    count_neg = y_train.value_counts()[0]
    count_pos = y_train.value_counts()[1]
    scale_pos_weight_value = count_neg / count_pos
    
    print(f"Calculado scale_pos_weight: {scale_pos_weight_value:.2f}")

    pipeline_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight_value,
            random_state=42
        ))
    ])
    
    
    param_dist = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [3, 5, 7, 9],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],         
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],  
        'model__gamma': [0, 0.1, 0.5, 1],               
        'model__min_child_weight': [1, 3, 5]            
    }
    
 
    random_search = RandomizedSearchCV(
        pipeline_xgb,
        param_distributions=param_dist,
        n_iter=50,  
        cv=5,       
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    
    print(f"Melhor score ROC AUC no CV: {random_search.best_score_:.4f}")
    print("Melhores hiperparâmetros encontrados:")
    print(random_search.best_params_)
    
    return random_search.best_estimator_

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

def build_and_tune_xgboost_multiclass(X_train, y_train):
    num_cols = ['ld', 'lq', 'resultado', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat', 'tpnascassi']
    cat_cols = ['parametro', 'grupo_de_parametros']
    
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])

    # Codificar rótulos e calcular pesos de amostra
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)

    # Pipeline completa
    pipeline_xgb_multi = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(
            objective='multi:softprob', num_class=len(le.classes_), eval_metric='mlogloss',
            random_state=42
        ))
    ])
    
    # Espaço de hiperparâmetros
    param_dist = {
        'model__n_estimators': [100, 200, 300], 'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1], 'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Busca randomizada
    random_search = RandomizedSearchCV(
        pipeline_xgb_multi, param_distributions=param_dist, n_iter=50, cv=5,
        scoring='f1_macro', n_jobs=-1, random_state=42, verbose=1
    )
    
    print("\nIniciando ajuste de hiperparâmetros para o XGBoost Multiclasse...")
    
    random_search.fit(X_train, y_train_encoded, model__sample_weight=sample_weights)

    print(f"Melhor score F1 Macro no CV: {random_search.best_score_:.4f}")
    print(f"Melhores hiperparâmetros: {random_search.best_params_}")
    
    return random_search.best_estimator_, le


def main():

    response_sinasc = requests.get(url_sinasc, params=params_url_sinasc, timeout=10)
    response_sisagua = requests.get(url_sisagua, params=params_url_sisagua, timeout=10)

    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()["parametros"])
    
    
    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunnatu', right_on='codigo_ibge', how='left')
    
    merged_df['idanomal'] = pd.to_numeric(merged_df['idanomal'], errors='coerce').fillna(0).astype(int)
    # Filtra os casos com informação válida sobre anomalias
    merged_df = merged_df[merged_df['idanomal'].isin([1, 2])].copy()
    merged_df['tem_anomalia'] = (merged_df['idanomal'] == 1).astype(int)
    merged_df['codanomal'] = merged_df['codanomal'].astype(str).fillna('0')
    
        
    features_index = ['ld','lq','resultado','grupo_de_parametros','parametro','idademae','gravidez','gestacao','consprenat','mesprenat','tpnascassi']
    

    X = merged_df[features_index]
    y = merged_df['tem_anomalia']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    binary_model_xgb = build_and_tune_xgboost(X_train, y_train)
    
    # Avalia o modelo final no treino e teste
    print("\n--- Avaliação do Modelo XGBoost Final ---")
    evaluate_binary(binary_model_xgb, X_train, y_train, prefix='treino binario')
    evaluate_binary(binary_model_xgb, X_test, y_test, prefix='teste binario')
    
    df_multi = merged_df[merged_df['tem_anomalia'] == 1].copy()
    
    # Agrupando classes raras para garantir que a estratificação funcione
    counts = df_multi['codanomal'].value_counts()
    rare = counts[counts < 10].index
    if not rare.empty:
        df_multi['codanomal_grouped'] = df_multi['codanomal'].replace(rare, 'Outros')
    else:
        df_multi['codanomal_grouped'] = df_multi['codanomal']
    
    X_multi = df_multi[features_index]
    y_multi = df_multi['codanomal_grouped']
    
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_multi, y_multi, test_size=0.25, stratify=y_multi, random_state=42)
    
    multi_model_xgb, label_encoder = build_and_tune_xgboost_multiclass(Xm_train, ym_train)
    
    # Avaliação do modelo multiclasse
    ym_train_pred_enc = multi_model_xgb.predict(Xm_train)
    ym_test_pred_enc  = multi_model_xgb.predict(Xm_test)

    # Decodificando os resultados para o formato original 
    ym_train_pred = label_encoder.inverse_transform(ym_train_pred_enc)
    ym_test_pred  = label_encoder.inverse_transform(ym_test_pred_enc)

    print("\n--- Avaliação Final do Modelo Multiclasse ---")
    print("[multiclass - TREINO] Classification report:")
    print(classification_report(ym_train, ym_train_pred))
    print("\n[multiclass - TESTE] Classification report:")
    print(classification_report(ym_test, ym_test_pred))

if __name__ == '__main__':
    main()
    