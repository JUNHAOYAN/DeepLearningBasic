import numpy as np
from Metric import iou
import torch
import torchvision
import time


def nms(detets, scores, threshold=0.6):
    """
    Non-maximum Suppression
    :param detets: bbox in shape [B, 4], [x1, y1, x2, y2]
    :param scores: bbox score in [B,]
    :param threshold: threshold
    :return: bbox after nms
    """

    keep = []

    # sorting from high 2 low
    scores_idx = np.argsort(scores, axis=0)[::-1]

    # get iou matrix
    iou_matrix = iou(detets[scores_idx, :], detets[scores_idx, :])

    while scores_idx.shape[0] > 0:
        max_ele_idx = scores_idx[0]
        keep.append(max_ele_idx)

        # delete max bbox score and iou of itself
        scores_idx = scores_idx[1:]
        bbox_iou = iou_matrix[0, 1:]
        iou_matrix = iou_matrix[1:, 1:]

        # find other bboxes where their iou are bigger than the threshold and remove them
        bbox_smaller_idx = bbox_iou <= threshold
        scores_idx = scores_idx[bbox_smaller_idx]
        iou_matrix = iou_matrix[:, bbox_smaller_idx]

    return keep


if __name__ == '__main__':
    bbox_1 = np.asarray([[3, 4, 6, 7], [0, 0, 4, 4], [0, 4, 6, 8], [7, 0, 9, 8], [0, 0, 3.5, 3.5]], dtype=np.float32)
    bbox_score = np.random.random([5]).astype(np.float32)
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
