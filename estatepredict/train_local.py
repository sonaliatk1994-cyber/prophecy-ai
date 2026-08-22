#!/usr/bin/env python3
"""
Local training script - no docker needed.
Run this after installing requirements.
"""
import sys
import os

# Add ml_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml_models"))

from lstm_model import train_lstm
from xgboost_model import train_xgboost

def main():
    data_path = "data/sample_properties.csv"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    print("="*60)
    print("EstatePredict - Local Model Training")
    print("="*60)

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found!")
        sys.exit(1)

    print("\n[1/2] Training LSTM models...")
    lstm_res = train_lstm(data_path, model_dir)

    print("\n[2/2] Training XGBoost classifier...")
    xgb_res = train_xgboost(data_path, model_dir)

    print("\n" + "="*60)
    print("DONE!")
    print(f"  LSTM areas trained: {len(lstm_res)}")
    print(f"  XGBoost accuracy: {xgb_res['accuracy']:.2%}")
    print(f"  Models saved to: {os.path.abspath(model_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()
