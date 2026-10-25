import numpy as np

def _fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def scores(label_trues, label_preds, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for (lt, lp) in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
    acc_cls = np.nanmean(acc_cls)
    denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iu = np.divide(np.diag(hist), denominator, out=np.zeros(num_classes), where=denominator > 0)
    valid = hist.sum(axis=1) > 0
    mean_iu = np.nanmean(iu[valid]) if valid.any() else 0.0
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))
    return {'Pixel Accuracy': acc, 'Mean Accuracy': acc_cls, 'Mean IoU': mean_iu, 'Class IoU': cls_iu}

def calculate_iou(pred, target, num_classes):
    iou_per_class = np.zeros(num_classes)
    valid_mask = target != 255
    target = target[valid_mask]
    pred = pred[valid_mask]
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union > 0:
            iou_per_class[cls] = intersection / union
        else:
            iou_per_class[cls] = np.nan
    return iou_per_class
