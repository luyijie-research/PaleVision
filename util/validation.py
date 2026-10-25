import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from . import eval_seg
from util import fid

def seg_validate_for_HSI1(backbone, seg_decoder, EVPL_model, criterion, data_loader, num_classes=10):
    val_loss = 0.0
    (preds, gts) = ([], [])
    backbone.eval()
    seg_decoder.eval()
    EVPL_model.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (hsi, rgb, labels, mask) = data
            outputs = seg_decoder(backbone(EVPL_model(hsi)))
            resized_outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            labels = labels.long().to(outputs.device)
            loss = criterion(resized_outputs, labels)
            val_loss += loss
            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
    score = eval_seg.scores(gts, preds, num_classes=num_classes)
    return (val_loss.cpu().numpy() / float(len(data_loader)), score)

def seg_validate_for_HSI2(backbone, seg_decoder, EVPL_model, criterion, data_loader, num_classes=19):
    val_loss = 0.0
    (preds, gts) = ([], [])
    backbone.eval()
    seg_decoder.eval()
    EVPL_model.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (hsi, rgb, labels, mask) = data
            outputs = seg_decoder(backbone(EVPL_model(hsi)))
            resized_outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            labels = labels.long().to(outputs.device)
            loss = criterion(resized_outputs, labels)
            val_loss += loss
            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
    score = eval_seg.scores(gts, preds, num_classes=num_classes)
    return (val_loss.cpu().numpy() / float(len(data_loader)), score)

def segmentation_validation(backbone, seg_decoder, EVPL_model, criterion, data_loader, num_classes):
    val_loss = 0.0
    total_iou_per_class = np.zeros(num_classes)
    valid_counts = np.zeros(num_classes)
    backbone.eval()
    seg_decoder.eval()
    EVPL_model.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (hsi, rgb, label, mask) = data
            labels = label.squeeze(1).type(torch.uint8)
            preds = seg_decoder(backbone(EVPL_model(hsi)))
            resized_outputs = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)
            labels = labels.long().to(preds.device)
            loss = criterion(resized_outputs, labels)
            val_loss += loss
            preds = torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16)
            labels = labels.cpu().numpy().astype(np.int16)
            for (pred, target) in zip(preds, labels):
                iou_per_class = eval_seg.calculate_iou(pred, target, num_classes)
                for cls in range(num_classes):
                    if not np.isnan(iou_per_class[cls]):
                        total_iou_per_class[cls] += iou_per_class[cls]
                        valid_counts[cls] += 1
            torch.cuda.empty_cache()
    avg_iou_per_class = np.divide(total_iou_per_class, valid_counts, where=valid_counts != 0)
    mean_iou = np.nanmean(avg_iou_per_class)
    return {'camouflage loss': val_loss.cpu().numpy() / float(len(data_loader)), 'Mean IoU': mean_iou}

def segmentation_validation_for_soil(backbone, seg_decoder, criterion, data_loader, num_classes):
    val_loss = 0.0
    total_iou_per_class = np.zeros(num_classes)
    valid_counts = np.zeros(num_classes)
    backbone.eval()
    seg_decoder.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (rgb, labels) = data
            preds = seg_decoder(backbone(rgb))
            resized_outputs = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)
            labels = labels.long().to(preds.device)
            loss = criterion(resized_outputs, labels)
            val_loss += loss
            preds = torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16)
            labels = labels.cpu().numpy().astype(np.int16)
            for (pred, target) in zip(preds, labels):
                iou_per_class = eval_seg.calculate_iou(pred, target, num_classes)
                for cls in range(num_classes):
                    if not np.isnan(iou_per_class[cls]):
                        total_iou_per_class[cls] += iou_per_class[cls]
                        valid_counts[cls] += 1
            torch.cuda.empty_cache()
    avg_iou_per_class = np.divide(total_iou_per_class, valid_counts, where=valid_counts != 0)
    mean_iou = np.nanmean(avg_iou_per_class)
    return {'seg loss': val_loss.cpu().numpy() / float(len(data_loader)), 'Mean IoU': mean_iou}

def cls_validate(model=None, data_loader=None):
    acc = 0.0
    model.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (inputs, labels) = data
            outputs = model(inputs)
            labels = labels.long().to(outputs.device)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels).sum().item()
    val_accurate = acc / len(data_loader)
    return val_accurate

def seg_validate_for_fr(backbone, seg_decoder, EVPL_model, criterion, data_loader, num_classes=19):
    val_loss = 0.0
    (preds, gts) = ([], [])
    backbone.eval()
    seg_decoder.eval()
    EVPL_model.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (hsi, labels) = data
            outputs = seg_decoder(backbone(EVPL_model(hsi)))
            resized_outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            labels = labels.long().to(outputs.device)
            loss = criterion(resized_outputs, labels)
            val_loss += loss
            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
    score = eval_seg.scores(gts, preds, num_classes=num_classes)
    return (val_loss.cpu().numpy() / float(len(data_loader)), score)

def segmentation_validation_fr(backbone, seg_decoder, EVPL_model, criterion, data_loader, num_classes):
    val_loss = 0.0
    total_iou_per_class = np.zeros(num_classes)
    valid_counts = np.zeros(num_classes)
    backbone.eval()
    seg_decoder.eval()
    EVPL_model.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (hsi, labels) = data
            preds = seg_decoder(backbone(EVPL_model(hsi)))
            resized_outputs = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)
            labels = labels.long().to(preds.device)
            loss = criterion(resized_outputs, labels)
            val_loss += loss
            preds = torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16)
            labels = labels.cpu().numpy().astype(np.int16)
            for (pred, target) in zip(preds, labels):
                iou_per_class = eval_seg.calculate_iou(pred, target, num_classes)
                for cls in range(num_classes):
                    if not np.isnan(iou_per_class[cls]):
                        total_iou_per_class[cls] += iou_per_class[cls]
                        valid_counts[cls] += 1
            torch.cuda.empty_cache()
    avg_iou_per_class = np.divide(total_iou_per_class, valid_counts, where=valid_counts != 0)
    mean_iou = np.nanmean(avg_iou_per_class)
    return {'camouflage loss': val_loss.cpu().numpy() / float(len(data_loader)), 'Mean IoU': mean_iou}

def segmentation_validation_modid(segmentor, criterion, data_loader, num_classes):
    val_loss = 0.0
    total_iou_per_class = np.zeros(num_classes)
    valid_counts = np.zeros(num_classes)
    device = next(segmentor.parameters()).device
    segmentor.eval()
    with torch.no_grad():
        for (_, data) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=' >='):
            (hsi, labels) = data
            hsi = hsi.to(device, non_blocking=True, dtype=torch.float32)
            labels = labels.to(device, non_blocking=True)
            preds = segmentor(hsi)
            resized_outputs = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)
            labels = labels.long().to(preds.device)
            loss = criterion(resized_outputs, labels)
            val_loss += loss
            preds = torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16)
            labels = labels.cpu().numpy().astype(np.int16)
            for (pred, target) in zip(preds, labels):
                iou_per_class = eval_seg.calculate_iou(pred, target, num_classes)
                for cls in range(num_classes):
                    if not np.isnan(iou_per_class[cls]):
                        total_iou_per_class[cls] += iou_per_class[cls]
                        valid_counts[cls] += 1
            torch.cuda.empty_cache()
    avg_iou_per_class = np.divide(total_iou_per_class, valid_counts, where=valid_counts != 0)
    mean_iou = np.nanmean(avg_iou_per_class)
    return {'camouflage loss': val_loss.cpu().numpy() / float(len(data_loader)), 'Mean IoU': mean_iou}
