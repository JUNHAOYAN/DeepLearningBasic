from abc import abstractmethod

import numpy as np


class Base:
    def __init__(self):
        self._cache = None

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray):
        pass

    def __str__(self):
        return "base class"


if __name__ == '__main__':
    print(Base())
