import os
import sys
from itertools import cycle
import datetime
import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.models as tvmodels
from dataloader.modid import ModidDataset
from models.backbone.mix_transformer import mit_b0, mit_b3, mit_b5, mit_b1
from models.HSI_to_RGB.modid_EVPL import EVPL
from models.decoder.segformer_head import SegFormerHead
from util import fid
from util.validation import segmentation_validation_fr
from util.optimizer import PolyWarmupAdamW
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/fish.yaml', type=str, help='config')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int, help='local_rank')
    parser.add_argument('--backend', default='nccl')
    parser.add_argument('--workdir', type=str, default='work_dir/camera')
    parser.add_argument('--backbone_pretrain', default='work_dir/Mars/backbone3_Epoch500.pth', help='the pretrained segmentor')
    parser.add_argument('--head_pretrain', default='work_dir/Mars/head3_Epoch500.pth', help='the pretrained segmentor')
    parser.add_argument('--evpl_pretrain', default='work_dir/Mars/EVPL3_Epoch500.pth', help='the pretrained evpl')
    parser.add_argument('--encoder_type', default='mit_b0', type=str, choices=['swin_b', 'swin_l', 'vit_b', 'vit_l', 'mit_b0', 'mit_b1', 'mit_b3', 'mit_b5'], help='Name of model to train')
    parser.add_argument('--seed', default=42, type=int)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['SLURM_PROCID']
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
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = time_now - time0
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return (str(delta), str(eta))

def train(cfg):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_workers = 4
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, timeout=datetime.timedelta(minutes=50))
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    num_classes = 2
    train_path = 'datasets/MODID'
    val_path = 'datasets/MODID'
    HSI_input_shape = [270, 510]
    HSI_train_dataset = ModidDataset(root_dir=train_path, input_shape=HSI_input_shape, train=True)
    HSI_val_dataset = ModidDataset(root_dir=val_path, input_shape=HSI_input_shape, train=False)
    HSI_train_sampler = DistributedSampler(HSI_train_dataset)
    HSI_train_loader = DataLoader(HSI_train_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=HSI_train_sampler, prefetch_factor=4)
    HSI_val_loader = DataLoader(HSI_val_dataset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    device = torch.device('cuda')
    print('%d GPUs are available:' % torch.cuda.device_count())
    (wavelength_1, wavelength_2, wavelength_3) = (600.0, 540.0, 460.0)

    def change_wavelength_to_channel(wavelength):
        return (wavelength - 450) / 4 + 1
    EVPL_model = EVPL(lamda_max_1=wavelength_1, lamda_max_2=wavelength_2, lamda_max_3=wavelength_3, device=device)
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
    backbone.to(device)
    backbone = DistributedDataParallel(backbone, device_ids=[args.local_rank], find_unused_parameters=True)
    seg_decoder = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=backbone.module.embed_dims, embedding_dim=256, num_classes=num_classes)
    seg_decoder.to(device)
    seg_decoder = DistributedDataParallel(seg_decoder, device_ids=[args.local_rank], find_unused_parameters=True)
    logging.info('successfully load pretrain model')
    beta = 0.1
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion = criterion.to(device)
    backbone_optimizer = torch.optim.Adam(backbone.parameters(), lr=6e-05)
    head_optimizer = torch.optim.SGD(seg_decoder.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    evpl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, EVPL_model.parameters()), lr=0.3)
    backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(backbone_optimizer, T_max=cfg.train.num_epoch)
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, T_max=cfg.train.num_epoch)
    evpl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(evpl_optimizer, T_max=cfg.train.num_epoch)
    for epoch in range(cfg.train.num_epoch):
        for (step, data) in enumerate(HSI_train_loader):
            backbone.train()
            seg_decoder.train()
            EVPL_model.train()
            (hsi, label) = data
            hsi = hsi.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            features = backbone(EVPL_model(hsi))
            seg_outputs = seg_decoder(features)
            seg_outputs = F.interpolate(seg_outputs, size=label.shape[1:], mode='bilinear', align_corners=False)
            seg_loss = criterion(seg_outputs, label.type(torch.long))
            head_optimizer.zero_grad()
            backbone_optimizer.zero_grad()
            evpl_optimizer.zero_grad()
            seg_loss.backward(retain_graph=True)
            backbone_optimizer.step()
            head_optimizer.step()
            evpl_optimizer.step()
            if args.local_rank == 0:
                (delta, eta) = cal_eta(time0, epoch + 1, cfg.train.num_epoch)
                logging.info('Epoch: %d;step: %d;Elasped: %s;eta: %s;seg_loss: %f' % (epoch + 1, step + 1, delta, eta, seg_loss.item()))
        backbone_scheduler.step()
        head_scheduler.step()
        for (name, param) in EVPL_model.module.filter_spectral_response.named_parameters():
            logging.info(f'Epoch {epoch + 1} - {name} weights: {param.data}')
        if (epoch + 1) % cfg.train.eval_epoch == 0:
            torch.save(seg_decoder.module.state_dict(), os.path.join(args.workdir, 'head_Epoch{}.pth'.format(epoch + 1)))
            torch.save(backbone.module.state_dict(), os.path.join(args.workdir, 'backbone_Epoch{}.pth'.format(epoch + 1)))
            torch.save(EVPL_model.module.state_dict(), os.path.join(args.workdir, 'evpl_Epoch{}.pth'.format(epoch + 1)))
            if args.local_rank == 0:
                logging.info('Validating...')
            val_accurate = segmentation_validation_fr(backbone=backbone, seg_decoder=seg_decoder, EVPL_model=EVPL_model, criterion=criterion, data_loader=HSI_val_loader, num_classes=num_classes)
            if args.local_rank == 0:
                logging.info(val_accurate)
    return True
file_path = os.path.abspath('__file__')
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
        log_dir = os.path.join(args.workdir, 'modid_exp.log')
        file = open(log_dir, 'w')
        file.close()
        setup_logger(filename=log_dir)
        logging.info('\nconfigs: %s' % cfg)
    setup_seed(1)
    train(cfg=cfg)
