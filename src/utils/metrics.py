import numpy as np


def compute_iou(pred, label):
    pred = np.argmax(pred, axis=0)
    classes = np.unique(label)
    ious = []
    for c in classes:
        pred_c = pred == c
        label_c = label == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0


def compute_iou_batch(preds, labels):
    iou = np.mean([compute_iou(pred, label) for pred, label in zip(preds, labels)])
    return iou
