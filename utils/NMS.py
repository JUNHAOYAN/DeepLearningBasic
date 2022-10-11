import numpy as np
# from Metric import iou
import torch
import torchvision
import time


def nms(dets: np.ndarray, scores: np.ndarray, threshold: float = 0.6):
    """
    Non-maximum Suppression
    :param dets: bbox in shape [B, 4], [x1, y1, x2, y2]
    :param scores: bbox score in [B,]
    :param threshold: threshold
    :return: bbox after nms
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    scores = np.argsort(scores)[::-1]

    keep = []

    while scores.shape[0]:
        # get the index with the highest confidence
        idx = scores[0]
        keep.append(idx)

        # calculate iou
        # left top coordinates
        x_lt = np.maximum(x1[idx], x1[scores[1:]])
        y_lt = np.maximum(y1[idx], y1[scores[1:]])
        # right bottom coordinates
        x_rb = np.minimum(x2[idx], x2[scores[1:]])
        y_rb = np.minimum(y2[idx], y2[scores[1:]])

        # intersection area
        inter = (x_rb - x_lt) * (y_rb - y_lt)
        # iou
        iou = inter / (areas[idx] + areas[scores[1:]] - inter)

        # get the index where the iou is smaller than the threshold
        idx_smaller_thresh = np.where(iou <= threshold)[0]

        # keep them and remove others
        scores = scores[idx_smaller_thresh + 1]

    return keep


if __name__ == '__main__':
    bbox_1 = np.asarray([[3, 4, 6, 7], [0, 0, 4, 4], [0, 4, 6, 8], [7, 0, 9, 8], [0, 0, 3.5, 3.5], [6, 7, 8, 8]],
                        dtype=np.float32)
    bbox_score = np.random.random([6]).astype(np.float32)
    start = time.time()
    print(nms(bbox_1, bbox_score, 0.6))
    end = time.time()
    print(end - start)

    a = torch.from_numpy(bbox_1)
    b = torch.from_numpy(bbox_score)
    start = time.time()
    print(torchvision.ops.nms(a, b, 0.6))
    end = time.time()
    print(end - start)
