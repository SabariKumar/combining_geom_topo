# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import multiprocessing
import pathlib
from typing import List
import os

if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
import pandas as pd
import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW as FusedAdam
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import sys
sys.path.append("/expanse/lustre/projects/slc154/sabarikumar/ProteinSol/combining_geom_topo")
from data_loading.topoformer.proteins import ProteinDataModule 
from model.topoformer.transformer import TopoformerPooled
from model.topoformer.fiber import Fiber
from model.topoformer.runtime import gpu_affinity
from model.topoformer.runtime.arguments import PARSER
from model.topoformer.runtime.callbacks import ProteinMetricCallback, QM9LRSchedulerCallback, BaseCallback, \
    PerformanceCallback, TestCorrectCountCallback
from runtime.topoformer.inference import evaluate_barcodes
from model.topoformer.runtime.loggers import LoggerCollection, DLLogger, WandbLogger, Logger
from model.topoformer.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, set_requires_grad

torch.autograd.set_detect_anomaly(True)
import warnings
warnings.filterwarnings('ignore', message='.*requires_grad.*')

FusedLAMB = None

def save_state(model: nn.Module, optimizer: Optimizer, epoch: int, path: pathlib.Path, callbacks: List[BaseCallback]):
    """ Saves model, optimizer and epoch states to path (only once per node) """
    if get_local_rank() == 0:
        state_dict = model.state_dict()
        state_dict_module = model.module.state_dict()
        checkpoint = {
            'state_dict': state_dict,
            'state_dict_module': state_dict_module,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        for callback in callbacks:
            callback.on_checkpoint_save(checkpoint)

        torch.save(checkpoint, str(path))
        #torch.save(model, str(path))
        logging.info(f'Saved checkpoint to {str(path)}')


def load_state(model: nn.Module, optimizer: Optimizer, path: pathlib.Path, callbacks: List[BaseCallback]):
    """ Loads model, optimizer and epoch states from path """
    checkpoint = torch.load(str(path), map_location={'cuda:0': f'cuda:{get_local_rank()}'})
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for callback in callbacks:
        callback.on_checkpoint_load(checkpoint)

    logging.info(f'Loaded checkpoint from {str(path)}')
    return checkpoint['epoch']


def train_epoch(model, train_dataloader, loss_fn, epoch_idx, grad_scaler, optimizer, local_rank, world_size, callbacks, args):
    loss_acc = torch.zeros((1,), device='cuda')
    loss_list = []
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='batch',
                         desc=f'Epoch {epoch_idx}', disable=(args.silent or local_rank != 0)):
        
        batch_list = list(batch)
        sids = batch_list.pop(0)
        batch = tuple(batch_list)
        *inputs, target = to_cuda(batch)
        #print(len(inputs))
        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            # try:
            #print(f'Working on sid {sids}')
            pred = model(*inputs)
            # except Exception as e:
            #     print(sids)
            #     print(e)
            #pred_sig = pred
            pred_sig = torch.sigmoid(pred)
            # pred_out = torch.zeros_like(pred_sig)
            # target_out = torch.zeros_like(target)
            # target_out = [torch.zeros_like(target) for i in range(world_size)]
            with torch.no_grad():
                pred_out = [torch.zeros_like(pred_sig) for _ in range(world_size)]
                torch.distributed.all_gather(pred_out, pred_sig)
                pred_out = torch.cat(pred_out)
                target_out = [torch.zeros_like(target) for _ in range(world_size)]
                torch.distributed.all_gather(target_out, target)
                target_out = torch.cat(target_out)
            #loss_dict = dict(zip(sids, zip(pred_sig.detach().cpu().numpy(), target.detach().cpu().numpy(), sids)))
            loss_dict = dict(zip(sids, zip(pred_out.detach().cpu().numpy(), target_out.detach().cpu().numpy(), sids)))
            loss_list.append(pd.DataFrame.from_dict(loss_dict, orient = 'index'))
            loss = loss_fn(pred, target) / args.accumulate_grad_batches
        loss_acc += loss.detach()
        grad_scaler.scale(loss).backward()

        # gradient accumulation
        if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad(set_to_none=True)

    if not loss_list:
        raise RuntimeError(
            f"Training dataloader yielded 0 batches in epoch {epoch_idx}. "
            f"Dataset is empty — check that PDB files and the CSV exist at the configured paths, "
            f"and that the DistributedSampler has samples for this rank."
        )
    loss_df = pd.concat(loss_list)
    loss_df.columns = ['pred', 'label', 'sids']

    return loss_df, loss_acc / (i + 1)

@torch.inference_mode()
def test_epoch(model, test_dataloader, local_rank, run_id, args):
    model.eval()
    loss_list = []
    for i, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader), unit = 'batch',
                         desc=f'Evaluating test', disable=(args.silent or local_rank != 0)):

        batch_list = list(batch)
        sids = batch_list.pop(0)
        batch = tuple(batch_list)
        *inputs, target = to_cuda(batch)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(*inputs)
            pred = torch.sigmoid(pred)
            loss_dict = dict(zip(sids, zip(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), sids)))
            loss_list.append(pd.DataFrame.from_dict(loss_dict, orient = 'index'))

    loss_df = pd.concat(loss_list)
    loss_df.columns = ['pred', 'label', 'sids']
    loss_df.to_csv(os.path.join(args.log_dir, f'{run_id}_test_losses.csv'))
    loss_df.to_pickle(os.path.join(args.log_dir, f'{run_id}_test_losses.pickle'))

    return loss_df

def train(model: nn.Module,
          loss_fn: _Loss,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          test_dataloader: DataLoader,
          callbacks: List[BaseCallback],
          logger: Logger,
          run_id: str,
          args):
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    logging.info(f'Is initialized: {is_distributed}')

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        logging.info('Running model in DDP mode')
        model._set_static_graph()

    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'lamb':
        if FusedLAMB is None:
            raise ImportError("LAMB optimizer requires NVIDIA Apex. Install apex or use --optimizer adam")
        optimizer = FusedLAMB(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    epoch_start = load_state(model, optimizer, args.load_ckpt_path, callbacks) if args.load_ckpt_path else 0

    for callback in callbacks:
        callback.on_fit_start(optimizer, args, epoch_start)

    logger.log_metrics({'train loss': 0}, -1)
    logger.log_grads(model)
    for epoch_idx in range(epoch_start, args.epochs):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        loss_df, loss = train_epoch(model, train_dataloader, loss_fn, epoch_idx, grad_scaler, optimizer, local_rank, world_size, callbacks,
                           args)
        if dist.is_initialized():
            torch.distributed.all_reduce(loss)
            loss /= world_size

        loss = loss.item()
        logging.info(f'Train loss: {loss}')
        logger.log_metrics({'train loss': loss}, epoch_idx)

        if epoch_idx + 1 == args.epochs:
            logger.log_metrics({'train loss': loss})

        for callback in callbacks:
            callback.on_epoch_end()

        if not args.benchmark and args.save_ckpt_path is not None and args.ckpt_interval > 0 \
                and (epoch_idx + 1) % args.ckpt_interval == 0:
            save_state(model, optimizer, epoch_idx, args.save_ckpt_path, callbacks)

        if not args.benchmark and (
                (args.eval_interval > 0 and (epoch_idx + 1) % args.eval_interval == 0) or epoch_idx + 1 == args.epochs):
            val_df = evaluate_barcodes(model, val_dataloader, callbacks, run_id, args)
            model.train()

            for callback in callbacks:
                callback.on_validation_end(epoch_idx)
            logger.log_table(val_df, 'val_preds')

        test_correct_callback = [TestCorrectCountCallback(logger)]
        evaluate_barcodes(model, test_dataloader, test_correct_callback, run_id, args)
        model.train()
        for callback in test_correct_callback:
            callback.on_validation_end(epoch_idx)

        logger.log_table(loss_df, 'train_preds')


    if args.save_ckpt_path is not None and not args.benchmark:
        save_state(model, optimizer, args.epochs, args.save_ckpt_path, callbacks)

    for callback in callbacks:
        callback.on_fit_end()

    test_callbacks = [ProteinMetricCallback(logger, targets_std=None, prefix='test')]
    test_loss = evaluate_barcodes(model, test_dataloader, test_callbacks, run_id, args)
    for callback in test_callbacks:
        callback.on_validation_end()
    logger.log_table(test_loss, 'test_preds')
    logger.log_artifact(test_loss, 'test_preds', run_id, pathlib.Path(args.log_dir))

def print_parameters_count(model):
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_params_trainable}')


if __name__ == '__main__':
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)

    logging.info('====== SE(3)-Transformer ======')
    logging.info('|      Training procedure     |')
    logging.info('===============================')

    #print(f"Fiber out dimensions {args.num_degrees * args.num_channels}")

    if args.seed is not None:
        logging.info(f'Using seed {args.seed}')
        seed_everything(args.seed)

    loggers = [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    if args.wandb:
        wandb_logger = WandbLogger(name=f'Topoformer', save_dir=args.log_dir, project='topoformer')
        #run_id = str(wandb_logger.experiment.id)
        loggers.append(wandb_logger)
    logger = LoggerCollection(loggers)

#Use with get_split_sizes_external
    # datamodule = ProteinDataModule(pdb_dir = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/filtered/train/pdbs',
    #                                sol_df = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/filtered/train/preprocessed/20250409_training_set.csv',
    #                                mode = 'train',
    #                                use_barcodes = True,  cache_processed = False, use_preprocessed = True,
    #                                processed_dir = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/filtered/train/preprocessed',
    #                                barcode_dir = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/filtered/train/rips_embeddings',
    #                                **vars(args))

    # datamodule = ProteinDataModule(pdb_dir = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/raw/train/pdbs',
    #                                sol_df = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/raw/csvs/20250324_training_set_no_errs.csv',
    #                                mode = 'train',
    #                                use_barcodes = True,  cache_processed = False, use_preprocessed = True,
    #                                processed_dir = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/train/preprocessed',
    #                                barcode_dir = '/home/sabari/ProteinSol/topoformer/data/soluprotgeom/processed/train/rips_embeddings/',
    #                                **vars(args))
    
    _repo_dir = os.environ.get('REPO_DIR', '/expanse/lustre/projects/slc154/sabarikumar/ProteinSol/combining_geom_topo')
    datamodule = ProteinDataModule(pdb_dir = os.path.join(_repo_dir, 'data/train'),
                                   sol_df = os.path.join(_repo_dir, 'data/csvs/training_set.csv'),
                                   mode = 'train',
                                   use_barcodes = True,
                                   processed_dir = os.path.join(_repo_dir, 'data/train'),
                                   barcode_dir = os.path.join(_repo_dir, 'data/train/'),
                                   external_test = os.path.join(_repo_dir, 'data/test'),
                                   external_df = os.path.join(_repo_dir, 'data/csvs/test_set.csv'),
                                   external_barcode_dir = os.path.join(_repo_dir, 'data/test/'),
                                   external_esm_dir = os.path.join(_repo_dir, 'data/test'),
                                   **vars(args))

    now = datetime.datetime.now()
    run_id = now.strftime("%Y%m%d%H%M")

    model = TopoformerPooled(
        output_dim=1,
        fiber_in = Fiber({0: datamodule.NODE_FEATURE_DIM}),
        fiber_out = Fiber({0: args.num_degrees * args.num_channels}),
        #num_heads = 1,
        fiber_edge = Fiber({0: datamodule.EDGE_FEATURE_DIM}),
        tensor_cores = using_tensor_cores(args.amp),
        comb_type = 'attn', #fctp or conv
        use_topo_projection = True,
        save_feats_dir = os.path.join(args.log_dir),
        run_id = run_id,
        **vars(args)
    )
    loss_fn = nn.BCEWithLogitsLoss()

    if args.benchmark:
        logging.info('Running benchmark mode')
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        callbacks = [PerformanceCallback(
            logger, args.batch_size * world_size, warmup_epochs=1 if args.epochs > 1 else 0
        )]
    else:
        callbacks = [ProteinMetricCallback(logger, targets_std=datamodule.targets_std, prefix='validation'),
                     QM9LRSchedulerCallback(logger, epochs=args.epochs)]

    if is_distributed:
        gpu_affinity.set_affinity(gpu_id=get_local_rank(), nproc_per_node=torch.cuda.device_count(), scope='socket')

    torch.set_float32_matmul_precision('high')
    print_parameters_count(model)
    logger.log_hyperparams(vars(args))
    increase_l2_fetch_granularity()
    train(model,
          loss_fn,
          datamodule.train_dataloader(),
          datamodule.val_dataloader(),
          datamodule.test_dataloader(),
          callbacks,
          logger,
          run_id,
          args)
    
    #print(model.transformer.last_layer_attention())

    logging.info('Training finished successfully')
    if args.wandb and (not dist.is_initialized() or dist.get_rank() == 0):
        import wandb
        wandb.finish()
