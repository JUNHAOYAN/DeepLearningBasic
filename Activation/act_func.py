"""
Activate function:
LeakyReLu, ReLu, Sigmoid, SoftMax, Tanh
"""
from .ActivationBaseClass import ActivationBase
import numpy as np


class Relu(ActivationBase):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x: np.ndarray):
        """
        :param x: in shape [B, C, H, W]
        :return: relu(x)
        """
        x = x.copy()
        self._cache["mask"] = x > 0

        # relu(x)
        x[~self._cache["mask"]] = 0

        return x

    def backward(self, dout: np.ndarray):
        """
        :param dout: same shape with the input x
        :return: dout * grad
        """
        assert self._cache is not None

        dout = dout.copy()

        return dout * self._cache["mask"]


class LeakyRelu(ActivationBase):
    def __init__(self, k=0.2):
        super(LeakyRelu, self).__init__()
        self.k = k

    def forward(self, x: np.ndarray):
        """
        :param x: in shape [B, C, H, W]
        :return: relu(x)
        """
        x = x.copy()
        self._cache["mask"] = x > 0

        # relu(x)
        x[~self._cache["mask"]] = x[~self._cache["mask"]] * self.k

        return x

    def backward(self, dout: np.ndarray):
        """
        :param dout: same shape with the input x
        :return: dout * grad
        """
        assert self._cache is not None

        d = np.ones_like(dout)
        d[~self._cache["mask"]] = d[~self._cache["mask"]] * self.k

        return d * dout


class Sigmoid(ActivationBase):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: np.ndarray):
        r"""
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
        :param x:
        :return:
        """
        clip_value = 1 / np.max(np.abs(x))
        x_clip = x * clip_value
        self._cache["k"] = clip_value

        sx = 1 / (1 + np.exp(-x_clip))
        self._cache["sx"] = sx

        return sx

    def backward(self, dout: np.ndarray):
        return self._cache["k"] * self._cache["sx"] * (1 - self._cache["sx"]) * dout


class Tanh(ActivationBase):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: np.ndarray):
        clip_value = 1 / np.max(np.abs(x))
        x_clip = x * clip_value
        self._cache["k"] = clip_value

        upper = np.exp(x_clip) - np.exp(-x_clip)
        lower = np.exp(x_clip) + np.exp(-x_clip)

        tanhx = upper / lower
        self._cache["tanhx"] = tanhx

        return tanhx

    def backward(self, dout: np.ndarray):
        return self._cache["k"] * (1 - self._cache["tanhx"] ** 2) * dout


class Softmax(ActivationBase):
    def __init__(self, axis=1):
        """
        :param axis: along which axis
        """
        super(Softmax, self).__init__()
        self.axis = axis

    def forward(self, x: np.ndarray):
        clip_value = 1 / np.max(np.abs(x), axis=self.axis, keepdims=True)
        x_clip = x * clip_value
        self._cache["k"] = clip_value

        expx = np.exp(x_clip)
        sumx = np.sum(expx, axis=self.axis, keepdims=True)
        softmaxx = expx / sumx

        return softmaxx

    def backward(self, dout: np.ndarray):
        # todo: combine with cross entropy
        pass


if __name__ == '__main__':
    a = np.random.randn(1, 3, 3, 3)
    relu = Softmax()
    out = relu.forward(a)

    dout = np.ones_like(a)
    out_backward = relu.backward(dout)
    print(out_backward)
