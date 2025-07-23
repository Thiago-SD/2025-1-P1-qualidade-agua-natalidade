import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer

url_sinasc = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-nascidos-vivos"
url_sisagua = "https://apidadosabertos.saude.gov.br/sisagua/controle-semestral"

params_url_sinasc = {
    "limit": 20000,  
    "offset": 0     
}

params_url_sisagua = {
    "limit": 20000,  
    "offset": 0,
    "semestre_de_referencia": 1     
}

def main():
    plot_dir = 'plots'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    response_sinasc = requests.get(url_sinasc, params=params_url_sinasc)
    response_sisagua = requests.get(url_sisagua, params=params_url_sisagua)
    
    df_sinasc = pd.DataFrame(response_sinasc.json()['sinasc'])
    df_sisagua = pd.DataFrame(response_sisagua.json()["parametros"])

    merged_df = pd.merge(df_sinasc, df_sisagua, left_on='codmunres', right_on='codigo_ibge', how='left')
    merged_df['idanomal'] = pd.to_numeric(merged_df['idanomal'], errors='coerce').astype(int)
    merged_df = merged_df[merged_df['idanomal'].isin([1,2])].copy()
    merged_df['tem_anomalia'] = (merged_df['idanomal'] == 1).astype(int)
    
    print("merged_df:", merged_df.shape)
    print("idanomal counts:\n", merged_df['idanomal'].value_counts(dropna=False))

    features_index = ['ld','lq','resultado','parametro','grupo_de_parametros', 'idademae','gravidez','gestacao','consprenat','mesprenat']
    X_bin = merged_df[features_index]
    y_bin = merged_df['tem_anomalia']

    # Divisão binária
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_bin, y_bin, 
        test_size=0.2, 
        stratify=y_bin, 
        random_state=42
    )
    
    df_multi = merged_df[merged_df['tem_anomalia'] == 1].copy()
    
    counts = df_multi['codanomal'].value_counts()
    rare_codes = counts[counts < 30].index.tolist()
    print(f"Códigos raros (n={len(rare_codes)}): {rare_codes}")
    
    df_multi.loc[:, 'codanomal'] = df_multi['codanomal'].apply(
        lambda x: 'Outros' if x in rare_codes else x
    )
    
    # Divisão multiclasse
    X_multi = df_multi[features_index]
    y_multi = df_multi['codanomal']
    
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(
        X_multi, y_multi,
        test_size=0.2,
        stratify=y_multi,
        random_state=42
    )
    
    # Obter categorias válidas
    categorias_validas = set(ym_train.unique())

    numeric_cols = ['ld', 'lq', 'resultado', 'idademae', 'gravidez', 'gestacao', 'consprenat', 'mesprenat']
    categorical_cols = ['parametro', 'grupo_de_parametros']

    # Transformador para colunas numéricas
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Transformador para colunas categóricas
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pré-processadores com imputação
    preprocessor_bin = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    preprocessor_multi = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    # Pipeline binário com SMOTE
    binary_pipeline_smote = ImbPipeline([
        ('prep', preprocessor_bin),
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    binary_pipeline_smote.fit(Xb_train, yb_train)

    # Pipeline multiclasse
    multi_pipeline = Pipeline([
        ('prep', preprocessor_multi),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    multi_pipeline.fit(Xm_train, ym_train)

    yb_pred = binary_pipeline_smote.predict(Xb_test)
    yb_proba = binary_pipeline_smote.predict_proba(Xb_test)[:, 1]
    
    print("\n>>> Modelo Binário - Classificação Anomalia vs Normal <<<")
    print(classification_report(yb_test, yb_pred, digits=4))
    print("ROC AUC:", roc_auc_score(yb_test, yb_proba))

    ym_pred = multi_pipeline.predict(Xm_test)
    
    print("\n>>> Modelo Multiclasse - Tipos de Anomalia <<<")
    print(classification_report(ym_test, ym_pred, digits=4))

    bin_pred = binary_pipeline_smote.predict(Xb_test)
    y_pred_full = pd.Series('SemAnomalia', index=Xb_test.index)
    
    anomaly_idx = bin_pred == 1
    if sum(anomaly_idx) > 0:
        Xb_test_anomaly = Xb_test[anomaly_idx]
        y_pred_multi = multi_pipeline.predict(Xb_test_anomaly)
        y_pred_full.iloc[anomaly_idx] = y_pred_multi

    y_true_full = merged_df.loc[Xb_test.index, 'codanomal'].copy()
    y_true_full.fillna('SemAnomalia', inplace=True)
    
    

    print("\n>>> Sistema Hierárquico - Predição Completa <<<")
    print(classification_report(y_true_full, y_pred_full, digits=4))
    
    # Matriz de confusão
    labels = sorted(set(y_true_full.unique()) | set(y_pred_full.unique()))
    cm = confusion_matrix(y_true_full, y_pred_full, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    print("Matriz de Confusão:")
    print(cm_df)

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_df, annot=True)

    plt.title('Mapa de Calor da Matriz de Confusão')
    plt.xlabel('Código Predito')
    plt.ylabel('Código Real')
    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(plot_dir + '/' + 'conf_matrix.png')
    else:
        plt.savefig('plots/' + 'conf_matrix.png')

if __name__ == "__main__":
    main()