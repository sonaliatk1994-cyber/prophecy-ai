import os
import sys

# add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lstm_model import train_lstm
from xgboost_model import train_xgboost

def main():
    data_path = "/app/data/sample_properties.csv"
    model_dir = "/app/models"

    os.makedirs(model_dir, exist_ok=True)

    print("="*50)
    print("Prophecy AI - Model Training Pipeline")
    print("="*50)

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found! Run data generator first.")
        sys.exit(1)

    print("\n[1/2] Training LSTM rent demand models...")
    lstm_results = train_lstm(data_path, model_dir)

    print("\n[2/2] Training XGBoost sale demand classifier...")
    xgb_results = train_xgboost(data_path, model_dir)

    print("\n" + "="*50)
    print("Training complete!")
    print(f"LSTM models: {len(lstm_results)} areas")
    print(f"XGBoost accuracy: {xgb_results['accuracy']:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()
