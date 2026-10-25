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
from dataloader.minneapple import MinneAppleHSIDataset
from models.backbone.mix_transformer import mit_b0, mit_b3, mit_b5, mit_b1
from models.HSI_to_RGB.minneapple_EVPL import EVPL, EVPLWithNormalization, ColorBlindEVPL
from models.decoder.segformer_head import SegFormerHead
from util.validation import segmentation_validation_fr
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser(description='MinneApple Segmentation Training')
    parser.add_argument('--config', default='configs/minneapple.yaml', type=str, help='')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int, help='local_rank')
    parser.add_argument('--backend', default='nccl', help='')
    parser.add_argument('--workdir', type=str, default='work_dir/minneapple', help='')
    parser.add_argument('--backbone_pretrain', default=None, help='backbone')
    parser.add_argument('--head_pretrain', default=None, help='head')
    parser.add_argument('--evpl_pretrain', default=None, help='EVPL')
    parser.add_argument('--encoder_type', default='mit_b0', type=str, choices=['mit_b0', 'mit_b1', 'mit_b3', 'mit_b5'], help='backbone')
    parser.add_argument('--seed', default=42, type=int, help='')
    parser.add_argument('--evpl_mode', default='passthrough', type=str, choices=['passthrough', 'color_transform', 'colorblind'], help='EVPL')
    parser.add_argument('--colorblind_type', default='protanopia', type=str, choices=['protanopia', 'deuteranopia', 'tritanopia', 'monochromacy'], help='(evpl_modecolorblind)')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.getenv('SLURM_PROCID', '0')
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
    fHandler = logging.FileHandler(filename, mode='w', encoding='utf-8')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)
    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = time_now - time0
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return (str(delta), str(eta))

def train(cfg, args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_workers = cfg.train.get('num_workers', 4)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, timeout=datetime.timedelta(minutes=50))
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    num_classes = cfg.dataset.num_classes
    data_path = cfg.dataset.root_dir
    input_shape = cfg.dataset.input_shape
    train_dataset = MinneAppleHSIDataset(root_dir=data_path, input_shape=input_shape, train=True, augment=True)
    val_dataset = MinneAppleHSIDataset(root_dir=data_path, input_shape=input_shape, train=False, augment=False)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.samples_per_gpu, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    device = torch.device('cuda')
    if args.local_rank == 0:
        logging.info(f'{torch.cuda.device_count()} GPUs are available')
    (wavelength_r, wavelength_g, wavelength_b) = (425.0, 540.0, 580.0)
    if args.evpl_mode == 'colorblind':
        EVPL_model = EVPLWithNormalization(lamda_max_1=wavelength_r, lamda_max_2=wavelength_g, lamda_max_3=wavelength_b, device=device, num_channels=31, use_isp=cfg.evpl.get('use_isp', False))
    else:
        EVPL_model = EVPL(lamda_max_1=wavelength_r, lamda_max_2=wavelength_g, lamda_max_3=wavelength_b, device=device, num_channels=31, use_isp=cfg.evpl.get('use_isp', False))
    if args.evpl_pretrain is not None:
        EVPL_model_dict = EVPL_model.state_dict()
        EVPL_pre_train_dict = torch.load(args.evpl_pretrain, map_location=device)
        EVPL_model_dict.update(EVPL_pre_train_dict)
        EVPL_model.load_state_dict(EVPL_model_dict)
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
        if args.local_rank == 0:
            logging.info(f'Loaded backbone pretrain from: {args.backbone_pretrain}')
    backbone.to(device)
    backbone = DistributedDataParallel(backbone, device_ids=[args.local_rank], find_unused_parameters=True)
    seg_decoder = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=backbone.module.embed_dims, embedding_dim=256, num_classes=num_classes)
    if args.head_pretrain is not None:
        seg_decoder_dict = seg_decoder.state_dict()
        seg_decoder_pre_train_dict = torch.load(args.head_pretrain, map_location=device)
        pretrain_dict = {k: v for (k, v) in seg_decoder_pre_train_dict.items() if k in seg_decoder_dict and v.shape == seg_decoder_dict[k].shape}
        seg_decoder_dict.update(pretrain_dict)
        seg_decoder.load_state_dict(seg_decoder_dict)
        if args.local_rank == 0:
            logging.info(f'Loaded head pretrain from: {args.head_pretrain}')
    seg_decoder.to(device)
    seg_decoder = DistributedDataParallel(seg_decoder, device_ids=[args.local_rank], find_unused_parameters=True)
    if args.local_rank == 0:
        logging.info('Models initialized successfully')
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion = criterion.to(device)
    backbone_optimizer = torch.optim.Adam(backbone.parameters(), lr=cfg.optimizer.learning_rate)
    head_optimizer = torch.optim.SGD(seg_decoder.parameters(), lr=cfg.head_optimizer.learning_rate, momentum=cfg.head_optimizer.momentum, weight_decay=cfg.head_optimizer.weight_decay)
    evpl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, EVPL_model.parameters()), lr=cfg.evpl_optimizer.learning_rate)
    backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(backbone_optimizer, T_max=cfg.train.num_epoch)
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, T_max=cfg.train.num_epoch)
    evpl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(evpl_optimizer, T_max=cfg.train.num_epoch)
    best_miou = 0.0
    for epoch in range(cfg.train.num_epoch):
        train_sampler.set_epoch(epoch)
        for (step, data) in enumerate(train_loader):
            backbone.train()
            seg_decoder.train()
            EVPL_model.train()
            (hsi, labels) = data
            print('hsi', hsi.shape)
            images = hsi.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features = backbone(EVPL_model(images))
            seg_outputs = seg_decoder(features)
            seg_outputs = F.interpolate(seg_outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            seg_loss = criterion(seg_outputs, labels.long())
            head_optimizer.zero_grad()
            backbone_optimizer.zero_grad()
            evpl_optimizer.zero_grad()
            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(EVPL_model.parameters(), max_norm=1.0)
            backbone_optimizer.step()
            head_optimizer.step()
            evpl_optimizer.step()
            if args.local_rank == 0:
                (delta, eta) = cal_eta(time0, epoch * len(train_loader) + step + 1, cfg.train.num_epoch * len(train_loader))
                logging.info(f'Epoch: {epoch + 1}; Step: {step + 1}/{len(train_loader)}; Elapsed: {delta}; ETA: {eta}; Loss: {seg_loss.item():.4f}')
        backbone_scheduler.step()
        head_scheduler.step()
        evpl_scheduler.step()
        if (epoch + 1) % cfg.train.eval_epoch == 0:
            if args.local_rank == 0:
                logging.info('Validating...')
            val_result = segmentation_validation_fr(backbone=backbone, seg_decoder=seg_decoder, EVPL_model=EVPL_model, criterion=criterion, data_loader=val_loader, num_classes=num_classes)
            if args.local_rank == 0:
                logging.info(f'Validation Result: {val_result}')
                current_miou = val_result['Mean IoU']
                if current_miou > best_miou:
                    best_miou = current_miou
                    torch.save(backbone.module.state_dict(), os.path.join(args.workdir, 'best_backbone.pth'))
                    torch.save(seg_decoder.module.state_dict(), os.path.join(args.workdir, 'best_head.pth'))
                    torch.save(EVPL_model.module.state_dict(), os.path.join(args.workdir, 'best_evpl.pth'))
                    logging.info(f'Saved best model with mIoU: {best_miou:.4f}')
                torch.save(backbone.module.state_dict(), os.path.join(args.workdir, f'backbone_Epoch{epoch + 1}.pth'))
                torch.save(seg_decoder.module.state_dict(), os.path.join(args.workdir, f'head_Epoch{epoch + 1}.pth'))
                torch.save(EVPL_model.module.state_dict(), os.path.join(args.workdir, f'evpl_Epoch{epoch + 1}.pth'))
    if args.local_rank == 0:
        logging.info(f'Training completed! Best mIoU: {best_miou:.4f}')
    return True
file_path = os.path.abspath(__file__)
cur_path = os.path.dirname(file_path)
project_path = os.path.dirname(cur_path)
sys.path.append(project_path)
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)
    if args.local_rank == 0:
        log_dir = os.path.join(args.workdir, 'minneapple_exp.log')
        setup_logger(filename=log_dir)
        logging.info(f'\nConfigs: {cfg}')
        logging.info(f'EVPL Mode: {args.evpl_mode}')
        if args.evpl_mode == 'colorblind':
            logging.info(f'Colorblind Type: {args.colorblind_type}')
    setup_seed(args.seed)
    train(cfg=cfg, args=args)
