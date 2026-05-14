# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Custom-first migration path for OpenMMLab 2.x
# ---------------------------------------------

from __future__ import division

import argparse
import copy
import importlib
import importlib.util
import os
import sys
import time
import torch
import warnings
from os import path as osp

from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import digit_version, import_modules_from_strings, mkdir_or_exist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import collect_env,register_all_modules
from mmengine.logging import MMLogger
REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true', help='automatically scale lr with the number of gpus')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot be both specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def import_plugin_modules(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return

    if hasattr(cfg, 'plugin_dir'):
        module_dir = cfg.plugin_dir
    else:
        module_dir = os.path.dirname(config_path)

    if module_dir.endswith('.py'):
        module_dir = os.path.dirname(module_dir)
    module_path = '.'.join([p for p in module_dir.split('/') if p])
    if module_path:
        importlib.import_module(module_path)

def build_model_compat(cfg):
    if 'train_cfg' in cfg or 'test_cfg' in cfg:
        return MODELS.build(cfg, default_args=dict(
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')))
    return MODELS.build(cfg)


def build_dataset_compat(cfg):
    return DATASETS.build(cfg)


def get_mmseg_version():
    if importlib.util.find_spec('mmseg') is None:
        return 'not installed'
    mmseg_module = importlib.import_module('mmseg')
    return getattr(mmseg_module, '__version__', 'unknown')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])

    import_plugin_modules(cfg, args.config)
    register_all_modules(init_default_scope=True)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        if osp.isfile(args.resume_from):
            cfg.resume_from = args.resume_from
            cfg.load_from = args.resume_from
            cfg.resume = True
        else:
            raise FileNotFoundError(f'--resume-from checkpoint not found: {args.resume_from}')

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # keep original AdamW workaround behavior
    if digit_version(torch.__version__) == digit_version('1.8.1'):
        if 'optimizer' in cfg and cfg.optimizer.get('type', None) == 'AdamW':
            cfg.optimizer['type'] = 'AdamW2'
        if ('optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper
                and cfg.optim_wrapper.optimizer.get('type', None) == 'AdamW'):
            cfg.optim_wrapper.optimizer['type'] = 'AdamW2'

    if args.autoscale_lr:
        if 'optimizer' in cfg and 'lr' in cfg.optimizer:
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
        elif 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper and 'lr' in cfg.optim_wrapper.optimizer:
            cfg.optim_wrapper.optimizer['lr'] = \
                cfg.optim_wrapper.optimizer['lr'] * len(cfg.gpu_ids) / 8
        elif 'auto_scale_lr' in cfg:
            cfg.auto_scale_lr.enable = True
        else:
            raise KeyError('Cannot autoscale lr: no optimizer/optim_wrapper lr found in cfg.')

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger_name = 'mmseg' if cfg.model.type in ['EncoderDecoder3D'] else 'mmdet'
    logger = MMLogger.get_instance(
        logger_name,
        log_level=cfg.log_level,
        log_file=log_file,
        file_mode='w'
    )

    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model_compat(cfg.model)
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset_compat(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset_compat(val_dataset))

    dataset_classes = getattr(datasets[0], 'CLASSES', None)
    if dataset_classes is None:
        # MMEngine-style dataset
        dataset_classes = datasets[0].metainfo.get('classes', None)

    #if cfg.checkpoint_config is not None:
        #mmseg_version = get_mmseg_version()
        #cfg.checkpoint_config.meta = dict(
            #mmdet_version=mmdet_version,
            #mmseg_version=mmseg_version,
            #mmdet3d_version=mmdet3d_version,
            #config=cfg.pretty_text,
            #CLASSES=dataset_classes,
            #PALETTE=datasets[0].PALETTE if hasattr(datasets[0], 'PALETTE') else None)

    model.CLASSES = dataset_classes

    # keep custom BEVFormer training path, migrate internals incrementally
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()