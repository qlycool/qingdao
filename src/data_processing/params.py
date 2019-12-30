import numpy as np


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


@Singleton
class Parameters(object):
    def __init__(self):
        base_path = '../../save',
        check_frequency = 2  # 2 sec
        sample_frequency = 2  # 2 sec
        update_frequency = 7  # 7 day

        self.base_path = base_path
        self.suggested_params = {
            'sample_frequency': sample_frequency,
            'check_frequency': check_frequency,
            'update_frequency': update_frequency
        }
        self.params = {
            'sample_frequency': sample_frequency,
            'check_frequency': check_frequency,
            'update_frequency': update_frequency
        }

    def update(self, key: str, value):
        if key in self.params.keys():
            self.params[key] = value

    def get(self, key: str):
        return self.params[key]

    def save(self, filename: str):
        if filename is None:
            raise Exception('Param saved path is None.')
        np.save(filename + '.npy', self.params)

    def load(self, filename: str):
        if filename is None:
            raise Exception('Param loaded path is None.')
        self.params = np.load(filename + '.npy')

    def reset(self):
        self.params = self.suggested_params
