import os

from src.config.config import MODEL_SAVE_DIR


def load_best_model_prefix(stage: str) -> str:
    all_model = []
    all_dir = load_all_model_dir()
    for each_dir in all_dir:
        for file in os.listdir(MODEL_SAVE_DIR + each_dir):
            if os.path.splitext(file)[0].split('#')[1] == stage:
                all_model.append(os.path.splitext(file)[0])

    all_mae = [str_.split('#')[2] for str_ in all_model]
    prefix = all_model[all_mae.index(min(all_mae))]
    return str(prefix.split('#')[0]) + '/' + prefix


def load_all_model_dir() -> list:
    return sorted(os.listdir(MODEL_SAVE_DIR), reverse=True)
