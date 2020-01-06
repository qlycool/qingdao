import numpy as np

from src.model.base import BasicModel


class HeadModel(BasicModel):
    def __init__(self, range_: int):
        self.init_per_brand = {}
        self.stable_per_brand = {}
        self.range_ = range_

    def train(self, init_per_brand: dict, stable_per_brand: dict):
        self.init_per_brand = init_per_brand
        self.stable_per_brand = stable_per_brand

    def save(self, saved_path: str):
        """
        Save model to disk
        :param saved_path: path and filename to save model and scaler
        """
        super().save(saved_path)
        self.check_model_state()
        f = open(saved_path + '.txt', 'w')
        head_model = {'init_per_brand': self.init_per_brand, 'stable_per_brand': self.stable_per_brand}
        f.write(str(head_model))
        f.close()

    def load(self, loaded_path: str):
        """
        load model from disk
        :param loaded_path: path and filename to load model and scaler
        """
        super().load(loaded_path)
        f = open(loaded_path + '.txt', 'r')
        head_model = eval(f.read())
        self.init_per_brand = head_model['init_per_brand']
        self.stable_per_brand = head_model['stable_per_brand']
        f.close()

    def check_model_state(self):
        if self.init_per_brand is None:
            raise Exception('No available init_per_brand.')
        if self.stable_per_brand is None:
            raise Exception('No available stable_per_brand.')

    def predict(self, brand: str, index: int) -> np.array:
        """
        predict in head stage
        :param brand: brand name
        :param index: index in head stage
        :return: predicted value
        """
        if index > self.range_:
            raise Exception('index is out of range. Might not in HEAD stage')
        # region 1
        if index < 16:  # magic number, generated after data processing
            pred_head_one = self.init_per_brand[brand][0]
        else:
            pred_head_one = self.stable_per_brand[brand][0]

        # region 2
        if index < 58:  # magic number, generated after data processing
            pred_head_two = self.init_per_brand[brand][1]
        else:
            pred_head_two = self.stable_per_brand[brand][1]

        return np.array([pred_head_one, pred_head_two]).T
