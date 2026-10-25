import os
import sys
import datetime
import argparse
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from dataloader.Mars import MarsDataset
from models.backbone.mix_transformer import mit_b0, mit_b3, mit_b5, mit_b1
from models.HSI_to_RGB.Mars3 import EVPL
from models.decoder.segformer_head import SegFormerHead
from util.validation import segmentation_validation
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/mars.yaml', type=str)
    parser.add_argument('--local_rank', default=int(os.getenv('LOCAL_RANK', 0)), type=int)
    parser.add_argument('--backend', default='nccl')
    parser.add_argument('--workdir', type=str, default='work_dir/mars')
    parser.add_argument('--train_path', type=str, default='datasets/Mars/MSL/train')
    parser.add_argument('--val_path', type=str, default='datasets/Mars/MSL/test')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--skip_val', action='store_true')
    parser.add_argument('--backbone_pretrain', default=None)
    parser.add_argument('--head_pretrain', default=None)
    parser.add_argument('--evpl_pretrain', default=None)
    parser.add_argument('--encoder_type', default='mit_b0', type=str, choices=['mit_b0', 'mit_b1', 'mit_b3', 'mit_b5'])
    parser.add_argument('--seed', default=42, type=int)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
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
    scale = (total_iter - cur_iter) / float(max(cur_iter, 1))
    delta = time_now - time0
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def maybe_load(module, path, device):
    if path is None or path == '':
        return
    state = torch.load(path, map_location=device)
    module.load_state_dict(state, strict=False)

def train(cfg):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_workers = args.num_workers
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, timeout=datetime.timedelta(minutes=50))
    time0 = datetime.datetime.now().replace(microsecond=0)
    num_classes = 9
    train_path = args.train_path
    val_path = args.val_path
    HSI_input_shape = [args.input_size, args.input_size]
    batch_size = args.batch_size if args.batch_size is not None else 24
    HSI_train_dataset = MarsDataset(root_dir=train_path, input_shape=HSI_input_shape, train=True)
    HSI_val_dataset = MarsDataset(root_dir=val_path, input_shape=HSI_input_shape, train=False)
    HSI_train_sampler = DistributedSampler(HSI_train_dataset)
    HSI_train_loader = DataLoader(HSI_train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=HSI_train_sampler, prefetch_factor=2)
    HSI_val_loader = DataLoader(HSI_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    device = torch.device('cuda', args.local_rank)
    if args.local_rank == 0:
        print('%d GPUs are available' % torch.cuda.device_count())
    wavelength_1, wavelength_2, wavelength_3 = 600.0, 540.0, 460.0
    EVPL_model = EVPL(lamda_max_1=wavelength_1, lamda_max_2=wavelength_2, lamda_max_3=wavelength_3, device=device)
    maybe_load(EVPL_model, args.evpl_pretrain, device)
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
    else:
        raise ValueError(args.encoder_type)
    maybe_load(backbone, args.backbone_pretrain, device)
    backbone.to(device)
    backbone = DistributedDataParallel(backbone, device_ids=[args.local_rank], find_unused_parameters=True)
    seg_decoder = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=backbone.module.embed_dims, embedding_dim=256, num_classes=num_classes)
    maybe_load(seg_decoder, args.head_pretrain, device)
    seg_decoder.to(device)
    seg_decoder = DistributedDataParallel(seg_decoder, device_ids=[args.local_rank], find_unused_parameters=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
    backbone_optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)
    head_optimizer = torch.optim.SGD(seg_decoder.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.0005)
    evpl_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, EVPL_model.parameters()), lr=2e-1)
    backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(backbone_optimizer, T_max=cfg.train.num_epoch)
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, T_max=cfg.train.num_epoch)
    evpl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(evpl_optimizer, T_max=cfg.train.num_epoch)
    global_step = 0
    for epoch in range(cfg.train.num_epoch):
        HSI_train_sampler.set_epoch(epoch)
        for step, data in enumerate(HSI_train_loader):
            backbone.train()
            seg_decoder.train()
            EVPL_model.train()
            hsi, rgb, label, mask = data
            hsi = hsi.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            features = backbone(EVPL_model(hsi))
            seg_outputs = seg_decoder(features)
            seg_outputs = F.interpolate(seg_outputs, size=label.shape[1:], mode='bilinear', align_corners=False)
            seg_loss = criterion(seg_outputs, label.type(torch.long))
            head_optimizer.zero_grad()
            backbone_optimizer.zero_grad()
            evpl_optimizer.zero_grad()
            seg_loss.backward()
            backbone_optimizer.step()
            head_optimizer.step()
            evpl_optimizer.step()
            global_step += 1
            if args.local_rank == 0:
                delta, eta = cal_eta(time0, epoch + 1, cfg.train.num_epoch)
                logging.info('Epoch: %d;step: %d;Elasped: %s;eta: %s;seg_loss: %f' % (epoch + 1, step + 1, delta, eta, seg_loss.item()))
            if args.max_steps is not None and global_step >= args.max_steps:
                if args.local_rank == 0:
                    logging.info('Reached max_steps=%d, stop training' % args.max_steps)
                dist.destroy_process_group()
                return True
        backbone_scheduler.step()
        head_scheduler.step()
        evpl_scheduler.step()
        if args.skip_val:
            continue
        train_accurate = segmentation_validation(backbone=backbone, seg_decoder=seg_decoder, EVPL_model=EVPL_model, criterion=criterion, data_loader=HSI_train_loader, num_classes=num_classes)
        if args.local_rank == 0:
            logging.info(train_accurate)
        for name, param in EVPL_model.module.filter_spectral_response.named_parameters():
            logging.info(f'Epoch {epoch + 1} - {name} weights: {param.data}')
        if (epoch + 1) % cfg.train.eval_epoch == 0:
            torch.save(EVPL_model.module.state_dict(), os.path.join(args.workdir, 'mars_rrbb_Epoch{}.pth'.format(epoch + 1)))
            if args.local_rank == 0:
                logging.info('Validating...')
            val_accurate = segmentation_validation(backbone=backbone, seg_decoder=seg_decoder, EVPL_model=EVPL_model, criterion=criterion, data_loader=HSI_val_loader, num_classes=num_classes)
            if args.local_rank == 0:
                logging.info(val_accurate)
    dist.destroy_process_group()
    return True

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)
    if args.local_rank == 0:
        log_dir = os.path.join(args.workdir, 'Mars_rrbb.log')
        open(log_dir, 'w').close()
        setup_logger(filename=log_dir)
        logging.info('\nconfigs: %s' % cfg)
    setup_seed(args.seed)
    train(cfg=cfg)
