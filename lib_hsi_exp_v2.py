import os
import sys
import datetime
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from dataloader.lib_hsi import LIBHSIDataset
from models.backbone.mix_transformer import mit_b0, mit_b3, mit_b5, mit_b1
from models.HSI_to_RGB.lib_hsi_EVPL import EVPL
from models.decoder.segformer_head import SegFormerHead
from util.validation import seg_validate_for_HSI2, segmentation_validation
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util import eval_seg

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            valid_mask = targets != self.ignore_index
            targets_clamped = targets.clone()
            targets_clamped[~valid_mask] = 0
            alpha_t = alpha.gather(0, targets_clamped.flatten())
            alpha_t = alpha_t.view(targets.shape)
            alpha_t[~valid_mask] = 0
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceLoss(nn.Module):

    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        valid_mask = (targets != self.ignore_index).unsqueeze(1).expand_as(targets_one_hot)
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0
        targets_one_hot.scatter_(1, targets_clamped.unsqueeze(1), 1)
        targets_one_hot = targets_one_hot * valid_mask.float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

def get_args_parser():
    parser = argparse.ArgumentParser(description='LIB-HSI Semantic Segmentation Training (Improved)')
    parser.add_argument('--config', default='configs/lib_hsi.yaml', type=str)
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--backend', default='nccl')
    parser.add_argument('--workdir', type=str, default='work_dir/lib_hsi_v2')
    parser.add_argument('--backbone_pretrain', default=None)
    parser.add_argument('--head_pretrain', default=None)
    parser.add_argument('--evpl_pretrain', default=None)
    parser.add_argument('--encoder_type', default='mit_b1', type=str, choices=['mit_b0', 'mit_b1', 'mit_b3', 'mit_b5'])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_classes', default=44, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help=' batch size (16) ')
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--loss_type', default='focal', type=str, choices=['ce', 'focal', 'ce_dice'], help='')
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--use_class_weight', action='store_true', default=True, help='')
    parser.add_argument('--backbone_lr', default=0.0001, type=float, help='backbone  ( head )')
    parser.add_argument('--head_lr', default=0.0005, type=float)
    parser.add_argument('--evpl_lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--grad_clip', default=1.0, type=float, help='')
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', default=0.2, type=float)
    parser.add_argument('--freeze_evpl', action='store_true', default=False, help=' EVPL ')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing factor (0.0-0.2 recommended)')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ.get('SLURM_PROCID', '0')
    return parser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)
    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now().replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter + 1e-06)
    delta = time_now - time0
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return (str(delta), str(eta))

def compute_class_weights(dataset, num_classes, device, max_weight=10.0):
    logging.info('Computing class weights from dataset...')
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    sample_size = min(100, len(dataset))
    indices = random.sample(range(len(dataset)), sample_size)
    for idx in indices:
        (_, label) = dataset[idx]
        label = label.numpy().flatten()
        label = label[label < num_classes]
        for c in label:
            class_counts[c] += 1
    total = class_counts.sum()
    weights = total / (class_counts + 1e-06)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, 0.1, max_weight)
    weights[class_counts == 0] = 1.0
    logging.info(f'Class weights computed. Min={weights.min():.2f}, Max={weights.max():.2f}')
    logging.info(f'Non-zero classes: {(class_counts > 0).sum().item()}/{num_classes}')
    return weights.float().to(device)

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    (y_a, y_b) = (y, y[index])
    return (mixed_x, y_a, y_b, lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-06):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train(cfg, args):
    setup_seed(args.seed)
    num_workers = 4
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, timeout=datetime.timedelta(minutes=50))
    time0 = datetime.datetime.now().replace(microsecond=0)
    device = torch.device('cuda')
    num_classes = args.num_classes
    train_path = 'datasets/LIB-HSI'
    val_path = 'datasets/LIB-HSI'
    HSI_input_shape = [args.input_size, args.input_size]
    HSI_train_dataset = LIBHSIDataset(root_dir=train_path, input_shape=HSI_input_shape, train=True, num_classes=num_classes)
    HSI_val_dataset = LIBHSIDataset(root_dir=val_path, input_shape=HSI_input_shape, train=False, num_classes=num_classes)
    HSI_train_sampler = DistributedSampler(HSI_train_dataset)
    HSI_train_loader = DataLoader(HSI_train_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=HSI_train_sampler, prefetch_factor=4)
    HSI_val_loader = DataLoader(HSI_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    if args.local_rank == 0:
        logging.info(f'{torch.cuda.device_count()} GPUs are available')
    (wavelength_1, wavelength_2, wavelength_3) = (450.0, 530.0, 580.0)
    EVPL_model = EVPL(lamda_max_1=wavelength_1, lamda_max_2=wavelength_2, lamda_max_3=wavelength_3, device=device, use_isp=False)
    if args.evpl_pretrain is not None:
        EVPL_model_dict = EVPL_model.state_dict()
        EVPL_pre_train_dict = torch.load(args.evpl_pretrain, map_location=device)
        EVPL_model_dict.update(EVPL_pre_train_dict)
        EVPL_model.load_state_dict(EVPL_model_dict)
        logging.info(f'Loaded EVPL pretrain from {args.evpl_pretrain}')
    if args.freeze_evpl:
        for param in EVPL_model.parameters():
            param.requires_grad = False
        logging.info('EVPL parameters frozen')
    EVPL_model.to(device)
    EVPL_model = DistributedDataParallel(EVPL_model, device_ids=[args.local_rank], find_unused_parameters=True)
    if args.encoder_type == 'mit_b0':
        backbone = mit_b0()
    elif args.encoder_type == 'mit_b1':
        backbone = mit_b1()
    elif args.encoder_type == 'mit_b3':
        backbone = mit_b3()
    elif args.encoder_type == 'mit_b5':
        backbone = mit_b5()
    if args.backbone_pretrain is not None:
        backbone_dict = backbone.state_dict()
        backbone_pre_train_dict = torch.load(args.backbone_pretrain, map_location=device)
        backbone_dict.update(backbone_pre_train_dict)
        backbone.load_state_dict(backbone_dict)
        logging.info(f'Loaded backbone pretrain from {args.backbone_pretrain}')
    backbone.to(device)
    backbone = DistributedDataParallel(backbone, device_ids=[args.local_rank], find_unused_parameters=True)
    seg_decoder = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=backbone.module.embed_dims, embedding_dim=256, num_classes=num_classes)
    if args.head_pretrain is not None:
        seg_decoder_dict = seg_decoder.state_dict()
        seg_decoder_pre_train_dict = torch.load(args.head_pretrain, map_location=device)
        seg_decoder_dict.update(seg_decoder_pre_train_dict)
        seg_decoder.load_state_dict(seg_decoder_dict)
        logging.info(f'Loaded decoder pretrain from {args.head_pretrain}')
    seg_decoder.to(device)
    seg_decoder = DistributedDataParallel(seg_decoder, device_ids=[args.local_rank], find_unused_parameters=True)
    class_weights = None
    if args.use_class_weight and args.local_rank == 0:
        class_weights = compute_class_weights(HSI_train_dataset, num_classes, device)
    if args.use_class_weight:
        if args.local_rank != 0:
            class_weights = torch.zeros(num_classes, device=device)
        dist.broadcast(class_weights, src=0)
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, ignore_index=255, label_smoothing=args.label_smoothing)
        logging.info(f'Using Focal Loss with gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}')
    elif args.loss_type == 'ce_dice':
        ce_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, label_smoothing=args.label_smoothing)
        dice_criterion = DiceLoss(num_classes=num_classes, ignore_index=255)
        criterion = lambda pred, target: ce_criterion(pred, target) + dice_criterion(pred, target)
        logging.info(f'Using CE + Dice Loss with label_smoothing={args.label_smoothing}')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, label_smoothing=args.label_smoothing)
        logging.info(f'Using CrossEntropy Loss with label_smoothing={args.label_smoothing}')
    backbone_optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.backbone_lr, weight_decay=args.weight_decay)
    head_optimizer = torch.optim.AdamW(seg_decoder.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    evpl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, EVPL_model.parameters()), lr=args.evpl_lr)
    total_epochs = cfg.train.num_epoch
    backbone_scheduler = WarmupCosineScheduler(backbone_optimizer, args.warmup_epochs, total_epochs, args.backbone_lr)
    head_scheduler = WarmupCosineScheduler(head_optimizer, args.warmup_epochs, total_epochs, args.head_lr)
    evpl_scheduler = WarmupCosineScheduler(evpl_optimizer, args.warmup_epochs, total_epochs, args.evpl_lr)
    if args.local_rank == 0:
        logging.info('=' * 60)
        logging.info('Training Configuration:')
        logging.info(f'  Backbone: {args.encoder_type}')
        logging.info(f'  Loss: {args.loss_type}')
        logging.info(f'  Batch size: {args.batch_size}')
        logging.info(f'  Input size: {args.input_size}')
        logging.info(f'  Backbone LR: {args.backbone_lr}')
        logging.info(f'  Head LR: {args.head_lr}')
        logging.info(f'  EVPL LR: {args.evpl_lr}')
        logging.info(f'  Warmup epochs: {args.warmup_epochs}')
        logging.info(f'  Class weights: {args.use_class_weight}')
        logging.info(f'  Freeze EVPL: {args.freeze_evpl}')
        logging.info('=' * 60)
    best_miou = 0.0
    for epoch in range(total_epochs):
        HSI_train_sampler.set_epoch(epoch)
        (preds, gts) = ([], [])
        epoch_loss = 0.0
        num_batches = 0
        for (step, data) in enumerate(HSI_train_loader):
            backbone.train()
            seg_decoder.train()
            EVPL_model.train()
            (hsi, label) = data
            print('hsi', hsi.shape)
            hsi = hsi.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if args.use_mixup:
                (hsi, label_a, label_b, lam) = mixup_data(hsi, label, args.mixup_alpha)
            rgb_features = EVPL_model(hsi)
            features = backbone(rgb_features)
            seg_outputs = seg_decoder(features)
            seg_outputs = F.interpolate(seg_outputs, size=label.shape[1:], mode='bilinear', align_corners=False)
            if args.use_mixup:
                seg_loss = mixup_criterion(criterion, seg_outputs, label_a.long(), label_b.long(), lam)
            else:
                seg_loss = criterion(seg_outputs, label.long())
            preds += list(torch.argmax(seg_outputs.detach(), dim=1).cpu().numpy().astype(np.int16))
            gts += list(label.detach().cpu().numpy().astype(np.int16))
            backbone_optimizer.zero_grad()
            head_optimizer.zero_grad()
            evpl_optimizer.zero_grad()
            seg_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(seg_decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(EVPL_model.parameters(), args.grad_clip)
            backbone_optimizer.step()
            head_optimizer.step()
            evpl_optimizer.step()
            epoch_loss += seg_loss.item()
            num_batches += 1
        backbone_lr = backbone_scheduler.step()
        head_scheduler.step()
        evpl_scheduler.step()
        score = eval_seg.scores(gts, preds, num_classes=num_classes)
        avg_loss = epoch_loss / num_batches
        if args.local_rank == 0:
            (delta, eta) = cal_eta(time0, epoch + 1, total_epochs)
            logging.info(f"Epoch: {epoch + 1}/{total_epochs}; Loss: {avg_loss:.4f}; Train mIoU: {score['Mean IoU']:.4f}; Elapsed: {delta}; ETA: {eta}")
            if not args.freeze_evpl:
                lambda_info = EVPL_model.module.get_lambda_max()
                logging.info(f"  Lambda: λ1={lambda_info['lambda_1']:.2f}nm, λ2={lambda_info['lambda_2']:.2f}nm, λ3={lambda_info['lambda_3']:.2f}nm (std={lambda_info['std']:.1f}nm fixed)")
            logging.info(f'  LR: backbone={backbone_lr:.2e}, head={head_scheduler.current_epoch}')
        if (epoch + 1) % cfg.train.eval_epoch == 0:
            backbone.eval()
            seg_decoder.eval()
            EVPL_model.eval()
            (val_preds, val_gts) = ([], [])
            val_loss_total = 0.0
            with torch.no_grad():
                for data in HSI_val_loader:
                    (hsi, label) = data
                    hsi = hsi.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    rgb_features = EVPL_model(hsi)
                    features = backbone(rgb_features)
                    seg_outputs = seg_decoder(features)
                    seg_outputs = F.interpolate(seg_outputs, size=label.shape[1:], mode='bilinear', align_corners=False)
                    val_loss = criterion(seg_outputs, label.long())
                    val_loss_total += val_loss.item()
                    val_preds += list(torch.argmax(seg_outputs, dim=1).cpu().numpy().astype(np.int16))
                    val_gts += list(label.cpu().numpy().astype(np.int16))
            val_score = eval_seg.scores(val_gts, val_preds, num_classes=num_classes)
            val_miou = val_score['Mean IoU']
            if args.local_rank == 0:
                logging.info(f"Validation: Loss={val_loss_total / len(HSI_val_loader):.4f}, mIoU={val_miou:.4f}, PA={val_score['Pixel Accuracy']:.4f}")
                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save(backbone.module.state_dict(), os.path.join(args.workdir, 'best_backbone.pth'))
                    torch.save(seg_decoder.module.state_dict(), os.path.join(args.workdir, 'best_head.pth'))
                    torch.save(EVPL_model.module.state_dict(), os.path.join(args.workdir, 'best_evpl.pth'))
                    logging.info(f'Best model saved! mIoU: {best_miou:.4f}')
        if args.local_rank == 0 and (epoch + 1) % 50 == 0:
            torch.save({'epoch': epoch + 1, 'backbone': backbone.module.state_dict(), 'head': seg_decoder.module.state_dict(), 'evpl': EVPL_model.module.state_dict(), 'backbone_optimizer': backbone_optimizer.state_dict(), 'head_optimizer': head_optimizer.state_dict(), 'evpl_optimizer': evpl_optimizer.state_dict(), 'best_miou': best_miou}, os.path.join(args.workdir, f'checkpoint_epoch{epoch + 1}.pth'))
    if args.local_rank == 0:
        logging.info(f'Training finished! Best validation mIoU: {best_miou:.4f}')
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    cfg = OmegaConf.load(args.config)
    if args.local_rank == 0:
        logging.info(f'\nConfigs: {cfg}')
        logging.info(f'Arguments: {args}')
    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir, exist_ok=True)
    setup_logger(os.path.join(args.workdir, 'lib_hsi_train_v2.log'))
    if args.local_rank == 0:
        logging.info(f'\nConfigs: {cfg}')
        logging.info(f'Arguments: {args}')
    train(cfg, args)
