import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class BasicModel:
    def save(self, saved_path: str):
        """
        Save model to disk
        :param saved_path: path and filename to save model and scaler
        """
        if saved_path is None:
            raise Exception('Saved path is None.')
        pass

    def load(self, loaded_path: str):
        """
        load model from disk
        :param loaded_path: path and filename to load model and scaler
        """
        if loaded_path is None:
            raise Exception('Loaded path is None.')
        pass

    def reset(self):
        """
        reset model
        """
        pass

    def check_model_state(self):
        """
        check model state before do some action
        """
        pass


class BasicLRModel(BasicModel):
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
        super().reset()
        """
        reset model
        """
        self.model = None
        self.scaler = None

    def check_model_state(self):
        super().check_model_state()
        if self.model is None:
            raise Exception('No available model, please train a new model or load from disk.')
        if self.scaler is None:
            raise Exception('No available scaler, please train a new model or load from disk.')

    # noinspection PyDictCreation
    @staticmethod
    def evaluate(y_true, y_pred, delta) -> dict:
        """
        evaluate the trained model
        :param y_true: ground truth label
        :param y_pred: predicted result
        :param delta: delta for computing mape
        :return: a dict for all performance metrics: region 1, region 2 and all
        """
        metrics = {}
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mae_1'] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        metrics['mae_2'] = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mse_1'] = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        metrics['mse_2'] = mean_squared_error(y_true[:, 1], y_pred[:, 1])
        if delta is not None:
            metrics['mape'] = BasicLRModel.mean_absolute_percentage_error(y_true, y_pred, delta)
            metrics['mape_1'] = BasicLRModel.mean_absolute_percentage_error(y_true[:, 0], y_pred[:, 0], delta[:, 0])
            metrics['mape_2'] = BasicLRModel.mean_absolute_percentage_error(y_true[:, 1], y_pred[:, 1], delta[:, 1])
        return metrics

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred, delta, delta_epsilon=1e-3) -> float:
        """
        compute mean_absolute_percentage_error based on delta
        :param y_true: ground truth label
        :param y_pred: predicted result
        :param delta: delta computed by y_true
        :param delta_epsilon:
        :return: mape
        """
        diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))
        delta = np.clip(np.abs(delta), delta_epsilon, None)
        return 100. * float(np.mean(diff / delta))
