from Activation import Sigmoid
from LossBaseClass import LossBase
import numpy as np


class L1Loss(LossBase):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, pred, target):
        assert len(pred.shape) == 4 and len(pred.shape) == len(target.shape)
        self._cache["pred_shape"] = pred.shape

        return np.mean(pred - target)

    def backward(self):
        b, c, h, w = self._cache["pred_shape"]

        return np.ones(self._cache["pred_shape"]) / (b * c * h * w)


class L2Loss(LossBase):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, pred, target):
        assert len(pred.shape) == 4 and len(pred.shape) == len(target.shape)
        self._cache["pred_shape"] = pred.shape
        diff = pred - target
        self._cache["diff"] = diff

        return np.mean(0.5 * diff ** 2)

    def backward(self):
        b, c, h, w = self._cache["pred_shape"]

        return self._cache["diff"] / (b * c * h * w)


class BinaryCrossEntropy(LossBase):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def __call__(self, pred, target):
        first = target * np.log(pred)
        second = (1 - target) * np.log(1 - pred)

        self._cache["x"] = pred
        self._cache["y"] = target
        self._cache["pred_shape"] = pred.shape

        return np.mean(first + second)

    def backward(self):
        b, c, h, w = self._cache["pred_shape"]
        n = b * c * h * w
        first = self._cache["y"] / self._cache["x"] / n
        second = (self._cache["y"] - 1) / (1 - self._cache["x"]) / n

        return first + second


class CrossEntropyWithSoftmax(LossBase):
    def __init__(self):
        super(CrossEntropyWithSoftmax, self).__init__()

    def __call__(self, pred, target):
        assert pred.shape[1] >= np.max(target)
        clip_value = 1 / np.max(np.abs(pred), axis=1, keepdims=True)
        x_clip = pred * clip_value
        self._cache["k"] = clip_value

        expx = np.exp(x_clip)
        sumx = np.sum(expx, axis=1, keepdims=True)
        softmaxx = expx / sumx

        self._cache["softmaxx"] = softmaxx
        self._cache["target"] = target

        # todo: target should be same size as softmaxx
        return np.log(softmaxx) * target

    def backward(self):
        return (self._cache["softmaxx"] - self._cache["target"]) * self._cache["k"]
