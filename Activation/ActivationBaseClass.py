from abc import abstractmethod
import numpy as np
from cache import Cache


class ActivationBase:
    def __init__(self):
        self._cache = Cache()

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray):
        pass

    def get_cache_keys(self):
        print(self._cache)