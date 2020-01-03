import os

from src.manager.model_manager import load_best_model_prefix
from src.data_processing.processing import read_data, data_clean, split_data_by_brand, generate_all_training_data, generate_validation_data, \
    predict_head, generate_head_dict
from src.model.lr_model import LRModel
from flask import Flask, jsonify, request
import pandas as pd
from src.config.config import MODEL_SAVE_DIR
from src.utils.util import create_dir, get_current_time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

model_produce = LRModel()
model_transition = LRModel()
init_per_brand = {}
stable_per_brand = {}


def load_current_model_prefix(stage: str) -> str:
    return load_best_model_prefix(stage)


@app.route('/api/load_current_model_name')
def api_load_current_model_name() -> str:
    return jsonify(load_current_model_prefix('produce'))


@app.route('/api/select_current_model_name', methods=['POST'])
def api_select_current_model_name():
    # TODO
    current_model_name = request.json['current_model_name']
    return current_model_name


@app.route('/api/train_val_model')
def api_train_val_model():
    # TODO
    model_save_dir = MODEL_SAVE_DIR + get_current_time()

    data = read_data('../data.csv')
    data = data_clean(data)
    data_per_brand, criterion = split_data_by_brand(data)
    X_produce, y_produce, delta_produce, mapping_produce = generate_all_training_data(data_per_brand, criterion, 'produce')
    X_transition, y_transition, delta_transition, mapping_transition = generate_all_training_data(data_per_brand, criterion, 'transition')

    metrics_produce = model_produce.train_validate(X_produce, y_produce, delta_produce, mapping_produce)
    metrics_transition = model_transition.train_validate(X_transition, y_transition, delta_transition, mapping_transition)

    # save model
    os.makedirs(model_save_dir)
    model_produce_save_path = model_save_dir + '#produce#' + metrics_produce['mae']
    model_transition_save_path = model_save_dir + '#transition#' + metrics_transition['mae']

    model_produce.save(model_produce_save_path)
    model_transition.save(model_transition_save_path)

    init_per_brand, stable_per_brand = generate_head_dict(data_per_brand, criterion)


def validation(data_per_batch: pd.DataFrame):
    test_produce, _ = generate_validation_data(data_per_batch, 'produce')
    test_transition, _ = generate_validation_data(data_per_batch, 'transition')

    pred_produce = model_produce.predict(test_produce)
    pred_transition = model_transition.predict(test_transition)
    pred_head = predict_head(data_per_batch, init_per_brand, stable_per_brand)


@app.route('/api/predict')
def api_predict():
    # TODO
    X_test = None
    return model_produce.predict(X_test)


if __name__ == '__main__':
    create_dir(MODEL_SAVE_DIR)
    app.run(host='0.0.0.0', debug=True)
