import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import os

def train_xgboost(csv_path="/app/data/sample_properties.csv", model_dir="/app/models"):
    print("[xgb] loading data...")
    df = pd.read_csv(csv_path)

    # feature engineering
    df['rent_per_sqft'] = df['rent_price_aed'] / df['sqft']
    df['sale_per_sqft'] = df['sale_price_aed'] / df['sqft']
    df['amenity_count'] = df['amenities'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)

    # encode categoricals
    df = pd.get_dummies(df, columns=['area', 'property_type'], prefix=['area', 'type'])

    # target: high sale demand (days_on_market < 20 = high demand)
    df['high_sale_demand'] = (df['days_on_market'] < 20).astype(int)

    # drop cols we dont need
    drop_cols = ['id', 'listing_date', 'timestamp', 'amenities', 'high_sale_demand']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['high_sale_demand']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # model params - tuned manually after some grid search attempts
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    print("[xgb] training...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"[xgb] accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # save
    model.save_model(f"{model_dir}/xgboost_model.json")

    # save feature names for inference
    with open(f"{model_dir}/xgb_features.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    # feature importance
    importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])

    with open(f"{model_dir}/xgb_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    results = {
        "accuracy": float(acc),
        "n_features": len(X.columns),
        "n_samples": len(df)
    }
    with open(f"{model_dir}/xgb_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[xgb] model saved to", model_dir)
    return results

if __name__ == "__main__":
    train_xgboost()
