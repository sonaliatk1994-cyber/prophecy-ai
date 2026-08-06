import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import json

def create_sequences(data, seq_length=10):
    """create sequences for LSTM - kinda hacky but works"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm(csv_path="/app/data/sample_properties.csv", model_dir="/app/models"):
    print("[lstm] loading data...")
    df = pd.read_csv(csv_path)

    # sort by area and synthetic date
    df['date'] = pd.to_datetime(df['listing_date'])
    df = df.sort_values(['area', 'date'])

    # group by area for time series
    all_preds = {}
    scalers = {}

    for area in df['area'].unique():
        area_df = df[df['area'] == area].copy()
        if len(area_df) < 50:
            continue  # skip small groups

        # feature: rent_per_sqft over time
        area_df = area_df.set_index('date').resample('D')['rent_price_aed'].mean().fillna(method='ffill')
        values = area_df.values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        scalers[area] = scaler

        seq_len = 7  # 7-day lookback
        if len(scaled) <= seq_len + 5:
            continue

        X, y = create_sequences(scaled, seq_len)

        # split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # build model - kept it simple for MSc project
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print(f"[lstm] training for {area}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=16,
            callbacks=[es],
            verbose=0
        )

        # predict next day
        last_seq = scaled[-seq_len:]
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)
        pred_actual = scaler.inverse_transform(pred)[0][0]

        all_preds[area] = {
            "predicted_rent": float(pred_actual),
            "rmse": float(np.sqrt(history.history['val_loss'][-1])),
            "samples": len(area_df)
        }

        # save model
        safe_name = area.replace(" ", "_").lower()
        model.save(f"{model_dir}/lstm_{safe_name}.keras")

    # save scalers
    with open(f"{model_dir}/lstm_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    with open(f"{model_dir}/lstm_results.json", "w") as f:
        json.dump(all_preds, f, indent=2)

    print("[lstm] done. results saved to", model_dir)
    return all_preds

if __name__ == "__main__":
    train_lstm()
