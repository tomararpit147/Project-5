"""
Model Training Script — Real Estate Investment Advisor
Run: python train_models.py
Requires: india_housing_prices.csv in the same directory
Outputs:  models/ directory with trained models + MLflow logs
"""
import pandas as pd 
import numpy as np
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor # type: ignore
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             classification_report, mean_squared_error,
                             mean_absolute_error, r2_score ) 

# Optional: MLflow
try:
    import mlflow 
    import mlflow.sklearn 
    MLFLOW = True
except ImportError:
    MLFLOW = False
    print("MLflow not installed — skipping experiment tracking. Install with: pip install mlflow")

os.makedirs('models', exist_ok=True)

# ── 1. LOAD & PREPROCESS ─────────────────────────────────────────────────────
print("Loading data...")
import os

import os
import requests

if not os.path.exists('india_housing_prices.csv'):
    print("Downloading dataset...")
    file_id = "1DenykRQDGQLUUKUMJbZSJ2cbHXI1jvgs"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    # Handle Google's virus scan warning for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"{url}&confirm={value}"
            response = session.get(url, stream=True)
            break
    with open('india_housing_prices.csv', 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print("Download complete!")

df = pd.read_csv('india_housing_prices.csv')
df.drop_duplicates(inplace=True)
print(f"Shape: {df.shape}")

# Feature engineering
df['Infra_Score'] = df['Nearby_Schools'] + df['Nearby_Hospitals']
df['Floor_Ratio'] = df['Floor_No'] / (df['Total_Floors'] + 1)
df['School_Density_Score'] = df['Nearby_Schools'] / (df['Size_in_SqFt'] / 1000 + 1)

# Target 1: Future Price (regression) — city-adjusted growth rate
city_growth = df.groupby('City')['Price_in_Lakhs'].median()
city_growth_rate = ((city_growth - city_growth.min()) /
                    (city_growth.max() - city_growth.min()) * 0.04) + 0.06
df['City_Growth_Rate'] = df['City'].map(city_growth_rate)
df['Future_Price_5yr'] = df['Price_in_Lakhs'] * (1 + df['City_Growth_Rate']) ** 5

# Target 2: Good Investment (classification)
median_ppsf = df['Price_per_SqFt'].median()
df['Good_Investment'] = (
    (df['Price_per_SqFt'] <= median_ppsf) &
    (df['BHK'] >= 2) &
    (df['Infra_Score'] >= 8)
).astype(int)
print(f"Good Investment ratio: {df['Good_Investment'].mean():.1%}")

# Encode categoricals
cat_cols = ['State','City','Locality','Property_Type','Furnished_Status',
            'Public_Transport_Accessibility','Parking_Space','Security',
            'Amenities','Facing','Owner_Type','Availability_Status']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col+'_enc'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
pickle.dump(le_dict, open('models/label_encoders.pkl','wb'))
print("Categoricals encoded.")

# ── 2. TRAIN/TEST SPLIT ──────────────────────────────────────────────────────
feature_cols = ['BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt','Age_of_Property',
                'Nearby_Schools','Nearby_Hospitals','Floor_No','Total_Floors',
                'Floor_Ratio','Infra_Score','School_Density_Score',
                'Property_Type_enc','Furnished_Status_enc','Public_Transport_Accessibility_enc',
                'Parking_Space_enc','Security_enc','Amenities_enc','Facing_enc',
                'Owner_Type_enc','Availability_Status_enc','State_enc','City_enc']

X = df[feature_cols]
y_cls = df['Good_Investment']
y_reg = df['Future_Price_5yr']

X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
pickle.dump(scaler, open('models/scaler.pkl','wb'))
pickle.dump(feature_cols, open('models/feature_cols.pkl','wb'))
print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

# ── 3. CLASSIFICATION MODELS ─────────────────────────────────────────────────
print("\n=== CLASSIFICATION ===")
cls_models = {
    'Logistic Regression': (LogisticRegression(max_iter=500, random_state=42), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1), False),
    'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42), False),
}
cls_results = {}

if MLFLOW:
    mlflow.set_experiment("Real_Estate_Classification")

for name, (model, use_scaled) in cls_models.items():
    X_train = X_tr_sc if use_scaled else X_tr
    X_test  = X_te_sc  if use_scaled else X_te
    
    run_ctx = mlflow.start_run(run_name=name) if MLFLOW else __import__('contextlib').nullcontext()
    with run_ctx:
        model.fit(X_train, yc_tr)
        pred = model.predict(X_test)
        acc = accuracy_score(yc_te, pred)
        f1  = f1_score(yc_te, pred)
        cls_results[name] = {'Accuracy': acc, 'F1': f1, 'model': model, 'scaled': use_scaled}
        print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        if MLFLOW:
            mlflow.log_params({'model': name, 'n_train': len(X_tr), 'n_test': len(X_te)})
            mlflow.log_metrics({'accuracy': acc, 'f1_score': f1})
            mlflow.sklearn.log_model(model, artifact_path="model")

best_cls_name = max(cls_results, key=lambda k: cls_results[k]['F1'])
best_cls_meta = cls_results[best_cls_name]
print(f"\n✅ Best Classifier: {best_cls_name} (F1={best_cls_meta['F1']:.4f})")
print(classification_report(yc_te,
      best_cls_meta['model'].predict(X_te_sc if best_cls_meta['scaled'] else X_te),
      target_names=['Not Good','Good Investment']))

pickle.dump(best_cls_meta['model'], open('models/classifier.pkl','wb'))
with open('models/best_classifier_name.txt','w') as f:
    f.write(best_cls_name)

# ── 4. REGRESSION MODELS ─────────────────────────────────────────────────────
print("\n=== REGRESSION ===")
reg_models = {
    'Ridge':             (Ridge(alpha=1.0), True),
    'Random Forest':     (RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1), False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42), False),
}
reg_results = {}

if MLFLOW:
    mlflow.set_experiment("Real_Estate_Regression")

for name, (model, use_scaled) in reg_models.items():
    X_train = X_tr_sc if use_scaled else X_tr
    X_test  = X_te_sc  if use_scaled else X_te
    
    run_ctx = mlflow.start_run(run_name=name) if MLFLOW else __import__('contextlib').nullcontext()
    with run_ctx:
        model.fit(X_train, yr_tr)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(yr_te, pred))
        mae  = mean_absolute_error(yr_te, pred)
        r2   = r2_score(yr_te, pred)
        reg_results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model': model, 'scaled': use_scaled}
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
        if MLFLOW:
            mlflow.log_params({'model': name})
            mlflow.log_metrics({'rmse': rmse, 'mae': mae, 'r2': r2})
            mlflow.sklearn.log_model(model, artifact_path="model")

best_reg_name = min(reg_results, key=lambda k: reg_results[k]['RMSE'])
best_reg_meta = reg_results[best_reg_name]
print(f"\n✅ Best Regressor: {best_reg_name} (RMSE={best_reg_meta['RMSE']:.2f}, R2={best_reg_meta['R2']:.4f})")

pickle.dump(best_reg_meta['model'], open('models/regressor.pkl','wb'))
with open('models/best_regressor_name.txt','w') as f:
    f.write(best_reg_name)

# ── 5. SAVE METRICS ──────────────────────────────────────────────────────────
summary = {
    'best_classifier': best_cls_name,
    'cls_accuracy': round(best_cls_meta['Accuracy'], 4),
    'cls_f1': round(best_cls_meta['F1'], 4),
    'best_regressor': best_reg_name,
    'reg_rmse': round(best_reg_meta['RMSE'], 2),
    'reg_mae': round(best_reg_meta['MAE'], 2),
    'reg_r2': round(best_reg_meta['R2'], 4),
    'all_classifiers': {k: {'Accuracy': round(v['Accuracy'],4), 'F1': round(v['F1'],4)}
                        for k,v in cls_results.items()},
    'all_regressors': {k: {'RMSE': round(v['RMSE'],2), 'MAE': round(v['MAE'],2), 'R2': round(v['R2'],4)}
                       for k,v in reg_results.items()},
}
json.dump(summary, open('models/metrics_summary.json','w'), indent=2)
print("\n✅ All models and metrics saved to ./models/")
print(json.dumps({k:v for k,v in summary.items() if 'all' not in k}, indent=2))
