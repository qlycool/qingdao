import os
from typing import Tuple

from config.config import ROOT_PATH, MODEL_SAVE_DIR, SCALER_SAVE_DIR


def load_all_model_scaler_prefix() -> list:
    return sorted([os.path.splitext(file)[0] for file in os.listdir(ROOT_PATH + MODEL_SAVE_DIR)], reverse=True)


def load_best_model_scaler_name() -> Tuple[str, str]:
    all_prefix = load_all_model_scaler_prefix()
    all_mae = [str_.split('#')[1] for str_ in all_prefix]
    min_index = all_mae.index(min(all_mae))
    prefix = all_prefix[min_index]
    return prefix + '.joblib', prefix + '.pkl'


def load_recent_model_scaler_name() -> Tuple[str, str]:
    prefix = load_all_model_scaler_prefix()[0]
    return prefix + '.joblib', prefix + '.pkl'
