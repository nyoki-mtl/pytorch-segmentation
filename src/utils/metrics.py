import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(preds, labels, classes=None):
    iou = np.nanmean([np.nanmean(compute_ious(pred, label, classes)) for pred, label in zip(preds, labels)])
    return iou


def iou_analyzer(preds, labels, tods):
    mIoU = np.nanmean([np.nanmean(compute_ious(pred, label, [1, 2, 3, 4])) for pred, label in zip(preds, labels)])
    print(f'Valid mIoU: {mIoU:.3f}\n')

    class_names = ['car', 'person', 'signal', 'road']
    tod_names = ['morning', 'day', 'night']
    iou_dict = {tod_name: dict(zip(class_names, [[] for _ in range(len(class_names))])) for tod_name in tod_names}
    for pred, label, tod in zip(preds, labels, tods):
        iou_per_class = compute_ious(pred, label, [1, 2, 3, 4])
        for iou, class_name in zip(iou_per_class, class_names):
            iou_dict[tod][class_name].append(iou)

    for tod_name in tod_names:
        print(f'\n---{tod_name}---')
        for k, v in iou_dict[tod_name].items():
            print(f'{k}: {np.nanmean(v):.3f}')

    print('\n---ALL---')
    for class_name in class_names:
        ious = []
        for tod_name in tod_names:
            ious += iou_dict[tod_name][class_name]
        print(f'{class_name}: {np.nanmean(ious):.3f}')
