import sys
import yaml
from unittest.mock import MagicMock

mock_bt = MagicMock()
sys.modules["bittensor"] = mock_bt

from miner_model_energy.pipeline import prepare_training_data, ModelConfig
from miner_model_energy.ml_config import ModelConfig

with open("model_params.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config = ModelConfig(**config_dict)

train_df, test_df, features = prepare_training_data(config, show_progress=True)

train_df.to_csv("local_research/data/training_data.csv", index=False)