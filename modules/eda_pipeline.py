# ✅ universal_eda_pipeline.py (Final Fix for IQR Outlier Removal)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np


def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data[vif_data['VIF'] > 5]['feature'].tolist()


def auto_eda_pipeline(df):
    eda_report = {}

    # Step 1: Drop ID-like columns
    drop_cols = [col for col in df.columns if 'id' in col.lower() or 'number' in col.lower()]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Step 2: Remove high-cardinality categoricals
    for col in df.select_dtypes(include='object'):
        if df[col].nunique() > 100:
            df.drop(columns=[col], inplace=True)

    # Step 3: Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Step 4: Detect target column
    target = None
    for col in df.columns[::-1]:
        if df[col].nunique() <= 10 and df[col].dtype in ['int64', 'float64']:
            target = col
            break
    if not target:
        target = df.columns[-1]

    y = df[target]
    X = df.drop(columns=[target])

    # Step 5: Smart Encoding
    for col in X.select_dtypes(include='object').columns:
        if X[col].nunique() <= 10:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Step 6: Chi-Square for categorical relevance (only if classification)
    chi2_removed = []
    if y.nunique() <= 10:
        for col in X.select_dtypes(include='uint8').columns:
            stat, p = chi2(X[[col]], y)
            if p[0] > 0.05:
                chi2_removed.append(col)
                X.drop(columns=[col], inplace=True)

    # Step 7: VIF for numeric features
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    high_vif = calculate_vif(X[numeric_cols])
    X.drop(columns=high_vif, inplace=True)

    # Step 8: Remove outliers using IQR (only numeric columns)
    numeric_only = X.select_dtypes(include=["float64", "int64"])
    Q1 = numeric_only.quantile(0.25)
    Q3 = numeric_only.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    is_outlier = ((numeric_only < lower_bound) | (numeric_only > upper_bound)).any(axis=1)
    non_outliers = ~is_outlier

    X_cleaned = X[non_outliers]
    y_cleaned = y[non_outliers]

    df_processed = pd.concat([X_cleaned, y_cleaned], axis=1)

    eda_report['outliers_removed'] = int(df.shape[0] - non_outliers.sum())
    eda_report['drop_cols'] = drop_cols
    eda_report['target'] = target
    eda_report['chi2_removed'] = chi2_removed
    eda_report['high_vif_removed'] = high_vif
    eda_report['shape_before'] = df.shape
    eda_report['shape_after'] = df_processed.shape
    eda_report['missing_handled'] = True

    return df_processed, eda_report