import time
import numpy as np
import torchvision
import torch


def iou(x: np.array, y: np.array):
    """
    calculate the iou of x and y
    :param x: in shape [B_1, 4], [[x1, y1, x2, y2], ...],
        where [x1, y1] is the left_top point, [x2, y2] is the right_bottom point
    :param y: in shape [B_2, 4], [[x1, y1, x2, y2], ...],
        where [x1, y1] is the left_top point, [x2, y2] is the right_bottom point
    :return: in shape [B1, B2]
    """
    area_x = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    area_y = (y[:, 2] - y[:, 0]) * (y[:, 3] - y[:, 1])

    overlap = [box_overlap(x, i) for i in y]
    overlap = np.stack(overlap, axis=1)

    return overlap / (area_x[:, np.newaxis] + area_y[np.newaxis, :] - overlap)


def box_overlap(x: np.array, y: np.array):
    """
    calculate the overlapping area of x and y
    :param x: in shape [B, 4], [[x1, y1, x2, y2], ...]
    :param y: in shape [, 4], [[x1, y1, x2, y2], ...]
    :return: in shape [B, 1]
    """
    if len(y.shape) == 1:
        y = y[np.newaxis, :]

    x_lt = np.maximum(x[:, 0], y[:, 0])
    y_lt = np.maximum(x[:, 1], y[:, 1])
    x_rb = np.minimum(x[:, 2], y[:, 2])
    y_rb = np.minimum(x[:, 3], y[:, 3])

    x_diff = x_rb - x_lt
    y_diff = y_rb - y_lt

    # clip
    x_diff = np.clip(x_diff, a_min=0, a_max=None)
    y_diff = np.clip(y_diff, a_min=0, a_max=None)

    return x_diff * y_diff


if __name__ == '__main__':
    bbox_1 = np.asarray([[3, 4, 6, 7], [0, 0, 4, 4], [1, 2, 7, 9]])
    bbox_2 = np.asarray([[3, 4, 6, 7], [5, 4, 6, 7]])

    start = time.time()
    result_torch = torchvision.ops.box_iou(torch.from_numpy(bbox_1), torch.from_numpy(bbox_2))
    end = time.time()
    print(end - start)

    start = time.time()
    result = iou(bbox_1, bbox_2)
    end = time.time()
    print(end - start)
