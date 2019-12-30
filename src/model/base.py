import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class BasicModel:
    def __init__(self):
        self.model = None
        self.scaler = None

    def train(self, X, y):
        pass

    def predict(self, X_test):
        self.check_model_state()
        if X_test is None or len(X_test) == 0:
            raise Exception('X_test is None or len == 1.')
        pass

    def save(self, model_saved_path: str, scaler_saved_name: str):
        """
        Save model to disk
        :param scaler_saved_name: path and filename to save scaler
        :param model_saved_path: path and filename to save model
        """
        self.check_model_state()
        if model_saved_path is None:
            raise Exception('Model saved path is None.')
        if scaler_saved_name is None:
            raise Exception('Scaler saved path is None.')
        pass

    def load(self, model_saved_path: str, scaler_saved_name: str):
        """
        load model from disk
        :param scaler_saved_name: path and filename to load scaler
        :param model_saved_path: path and filename to load
        """
        if model_saved_path is None:
            raise Exception('Model loaded path is None.')
        if scaler_saved_name is None:
            raise Exception('Scaler loaded path is None.')
        pass

    def train_validate(self, X, y, delta, test_size=0.2):
        """
        Train and val this model for evaluation
        :param X: train and val data
        :param y: train and val label
        :param delta: the delta value
        :param test_size: split ratio for validation set
        """
        pass

    def reset(self):
        """
        reset model
        """
        self.model = None
        self.scaler = None

    def check_model_state(self):
        if self.model is None:
            raise Exception('No available model, please train a new model or load from disk.')
        if self.scaler is None:
            raise Exception('No available scaler, please train a new model or load from disk.')

    @staticmethod
    def evaluate(y_true, y_pred, delta) -> dict:
        metrics = {}
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        if delta is not None:
            metrics['mape'] = BasicModel.mean_absolute_percentage_error(y_true, y_pred, delta)
        return metrics

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred, delta, delta_epsilon=1e-3) -> float:
        diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))
        delta = np.clip(np.abs(delta), delta_epsilon, None)
        return 100. * float(np.mean(diff / delta))
