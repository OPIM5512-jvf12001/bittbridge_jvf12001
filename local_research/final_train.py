import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DML_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import types


# 1. Create a fake termios module
termios_mock = types.ModuleType('termios')
termios_mock.TIOCGWINSZ = 0x5413  # The hex code Alembic is looking for
termios_mock.TCSAFLUSH = 2
sys.modules['termios'] = termios_mock

# 2. Create a fake fcntl module
fcntl_mock = types.ModuleType('fcntl')
fcntl_mock.ioctl = lambda fd, op, arg=0: b'\x00' * 8  # Returns a dummy 8-byte string
sys.modules['fcntl'] = fcntl_mock

import os

# 1. FORCE RESET: Remove 'fork' from the OS module if it was added
if hasattr(os, 'fork'):
    del os.fork
if hasattr(os, 'register_at_fork'):
    del os.register_at_fork

# 2. MODULE PURGE: Force Python to forget the 'broken' threadpool it tried to load
for mod in ['concurrent', 'concurrent.futures', 'concurrent.futures.thread', 'concurrent.futures._base']:
    if mod in sys.modules:
        del sys.modules[mod]


import concurrent.futures.thread
import concurrent.futures.process

import concurrent.futures
concurrent.futures.ThreadPoolExecutor = concurrent.futures.thread.ThreadPoolExecutor
concurrent.futures.ProcessPoolExecutor = concurrent.futures.process.ProcessPoolExecutor

sys.modules['concurrent.futures'] = concurrent.futures
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import json

from miner_model_energy.pipeline import prepare_training_data
from miner_model_energy.ml_config import load_model_config

config = load_model_config("model_params.yaml")

train_df, test_df, feature_cols = prepare_training_data(config, show_progress=True)

# We combine train and test because for the final model, we want the most data possible
df_final = pd.concat([train_df, test_df])
target_col = "target_load_horizon"

df_final = df_final.dropna(subset=[target_col] + feature_cols)
df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

X_raw = df_final[feature_cols].values
y_raw = df_final[target_col].values

artifact_dir = Path("best_model_artifacts")
artifact_dir.mkdir(exist_ok=True)

# Create and Save Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
scaler_path = artifact_dir / "lstm_input_scaler.joblib"
joblib.dump(scaler, scaler_path)

X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        96, 
        input_shape=(1, len(feature_cols)),
        activation='tanh',      
        recurrent_activation='sigmoid', 
        unroll=True                  
    ),
    tf.keras.layers.Dropout(0.3233986436474498),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0004893452546721716), loss='mse', metrics=['mape'])
model.fit(X_lstm, y_raw, epochs=50, batch_size=32, verbose=1)

# Save the model file
model_path = artifact_dir / "model.h5"
model.save(model_path)

# Create manifest
manifest = {
    "model_type": "lstm",
    "features": feature_cols,
    "lstm_n_steps": 1,
    "lstm_standardize_inputs": True,
    "lstm_scaler_path": "lstm_input_scaler.joblib",
    "model_path": "model.h5"
}


with open(artifact_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)

print(f"Compatibility Bundle Created in: {artifact_dir}")