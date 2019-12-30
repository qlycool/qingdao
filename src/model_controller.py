from manager.model_manager import load_all_model_scaler_prefix, load_best_model_scaler_name
from src.data_processing.processing import read_data, data_clean, split_data_by_brand, generate_all_training_data
from src.model.lr_model import LRModel
from flask import Flask, jsonify, request
from typing import Tuple

from config.config import ROOT_PATH, MODEL_SAVE_DIR, SCALER_SAVE_DIR
from src.utils.util import create_dir, get_current_time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

model = LRModel()


def load_current_model_name() -> Tuple[str, str]:
    # TODO
    model_name, scaler_name = load_best_model_scaler_name()
    return model_name, scaler_name


@app.route('/api/load_all_model_name')
def api_load_all_model_name():
    return jsonify(load_all_model_scaler_prefix())


@app.route('/api/load_current_model_name')
def api_load_current_model_name() -> str:
    return jsonify(load_current_model_name())


@app.route('/api/select_current_model_name', methods=['POST'])
def api_select_current_model_name():
    # TODO
    current_model_name = request.json['current_model_name']
    return current_model_name


@app.route('/api/train_model')
def api_train_model():
    # TODO
    data = read_data('../data.csv')
    data = data_clean(data)
    data_per_brand, criterion = split_data_by_brand(data)
    X, y, _, _ = generate_all_training_data(data_per_brand, criterion)

    model.train(X, y)
    current = get_current_time()
    model.save(ROOT_PATH + MODEL_SAVE_DIR + current + '.joblib', ROOT_PATH + SCALER_SAVE_DIR + current + '.pkl')


@app.route('/api/predict')
def api_predict():
    # TODO
    X_test = None
    return model.predict(X_test)


if __name__ == '__main__':
    create_dir(ROOT_PATH + MODEL_SAVE_DIR)
    create_dir(ROOT_PATH + SCALER_SAVE_DIR)
    app.run(host='0.0.0.0', debug=True)
