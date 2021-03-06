from src.model.base import BasicLRModel
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


class LRModel(BasicLRModel):
    def __init__(self, standardize=True):
        super().__init__()
        self.standardize = standardize
        self.scaler = StandardScaler()
        self.mapping_test = None

    def train(self, X, y):
        """
        Train a model with train data X and label y
        :param X: train data, shape = (Sample number, Feature number)
        :param y: train label, shape = (Sample number, 2): temperature of region 1 and temperature of region 2
        """
        super().train(X, y)
        self.model = LinearRegression()
        if self.standardize:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X_test: np.array or list) -> np.array:
        """
        Predict using trained model
        :param X_test: test data, shape = (Sample number, Feature number)
        :return: a array with exactly 2 number: temperature of region 1 and temperature of region 2
        """
        super().predict(X_test)
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape((1, len(X_test)))

        if self.standardize:
            X_test = self.scaler.transform(X_test)

        pred = self.model.predict(X_test)
        return pred

    def save(self, saved_path: str):
        super().save(saved_path)
        dump(self.model, saved_path + '.joblib')
        dump(self.scaler, saved_path + '.pkl')

    def load(self, loaded_path: str):
        super().load(loaded_path)
        self.model = load(loaded_path + '.joblib')
        self.scaler = load(loaded_path + '.pkl')

    def train_validate(self, X, y, delta=None, mapping=None, test_size=0.2) -> dict:
        super().train_validate(X, y, delta, test_size)
        X_train, X_test, y_train, y_test, delta_train, delta_test, mapping_train, mapping_test = None, None, None, None, None, None, None, None
        if delta is None and mapping is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=6)
        elif mapping is None:
            X_train, X_test, y_train, y_test, delta_train, delta_test = train_test_split(X, y, delta, test_size=test_size, random_state=6)
        elif delta is None:
            X_train, X_test, y_train, y_test, mapping_train, mapping_test = train_test_split(X, y, mapping, test_size=test_size, random_state=6)
        else:
            X_train, X_test, \
            y_train, y_test, \
            delta_train, delta_test, \
            mapping_train, mapping_test = train_test_split(X,
                                                           y,
                                                           delta,
                                                           mapping,
                                                           test_size=test_size,
                                                           random_state=6)

        self.mapping_test = mapping_test
        self.train(X_train, y_train)
        pred = self.predict(X_test)
        return super().evaluate(y_test, pred, delta_test)
