from abc import abstractmethod

from cache import Cache


class LossBase:
    def __init__(self):
        self._cache = Cache()

    @abstractmethod
    def __call__(self, pred, target):
        pass

    @abstractmethod
    def backward(self):
        pass

    def get_cache_keys(self):
        print(self._cache)
