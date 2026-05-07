import sys
import types

# 1. Create a fake termios module
termios_mock = types.ModuleType('termios')
termios_mock.TIOCGWINSZ = 0x5413  # The hex code Alembic is looking for
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["tf_device"] = "cpu"

import concurrent.futures.thread
import concurrent.futures.process

import concurrent.futures
concurrent.futures.ThreadPoolExecutor = concurrent.futures.thread.ThreadPoolExecutor
concurrent.futures.ProcessPoolExecutor = concurrent.futures.process.ProcessPoolExecutor

sys.modules['concurrent.futures'] = concurrent.futures

print(f"🛠️ Manual Repair: ThreadPoolExecutor is {getattr(concurrent.futures, 'ThreadPoolExecutor', 'STILL MISSING')}")

import yaml
import logging
from pathlib import Path
import tensorflow as tf
import optuna
import gc


tf.config.set_visible_devices([], 'GPU')


from miner_model_energy.pipeline import train_model
from miner_model_energy.ml_config import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def objective(trial):
    config_path = Path("model_params.yaml")
    with open(config_path, "r") as f:
        base_config_dict = yaml.safe_load(f)
    
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    # Capping units to avoid the risk of filling up VRAM on a long search
    units = trial.suggest_int("units", 32, 128, step=32)
    dense_units = trial.suggest_int("dense_units", 16, 64, step=16)
    
    # High dropout to help avoid overfitting due to added features
    dropout = trial.suggest_float("dropout", 0.2, 0.5) 
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    base_config_dict['models']['lstm']['learning_rate'] = lr
    base_config_dict['models']['lstm']['units'] = units
    base_config_dict['models']['lstm']['dense_units'] = dense_units
    base_config_dict['models']['lstm']['dropout'] = dropout
    base_config_dict['models']['lstm']['batch_size'] = batch_size

    base_config_dict['models']['lstm']['standardize_inputs'] = True
    base_config_dict['models']['lstm']['early_stopping_patience'] = 8
    base_config_dict['models']['lstm']['epochs'] = 100

    try:
        logging.info(f"Starting Trial {trial.number}: units={units}, lr={lr:.5f}, dropout={dropout:.2f}")

        config_obj = ModelConfig(**base_config_dict)  
        result = train_model("lstm", config_obj)
        
        # Return the validation MAPE
        val_mape = result.metrics['validation']['mape']
        logging.info(f"Trial {trial.number} finished with MAPE: {val_mape:.4f}")
        return val_mape
        
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {e}")
        return float('inf')

    finally:
        # This clears the GPU memory and resets the graph names
        tf.keras.backend.clear_session()
        # Force garbage collection for good measure
        gc.collect()

if __name__ == "__main__":

    # Setting a time limit to run the search overnight and let Optuna figure out the rest
    TIME_LIMIT_SECONDS = 25200 

    print("Starting Overnight Optuna Search")
    print(f"Time limit set to {TIME_LIMIT_SECONDS / 3600:.1f} hours.")
    
    # The SQLite database saves every trial to immediately just in case a crash happens, the trial data is safe.
    study = optuna.create_study(
        study_name="overnight_lstm_run",
        storage="sqlite:///optuna_overnight.db", 
        direction="minimize",
        load_if_exists=True
    )
    
    # Run the optimization with the strict time limit
    study.optimize(objective, timeout=TIME_LIMIT_SECONDS)

    print("\n" + "="*50)
    print("Best Parameters Found:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")
    print(f"Best Validation MAPE: {study.best_value:.4f}")
    print("="*50)