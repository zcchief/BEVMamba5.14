#  Custom-first migration path for OpenMMLab 2.x
# ---------------------------------------------
import argparse
import importlib
import os
import os.path as osp
import time
import warnings
import copy
import torch
from mmcv.cnn import fuse_conv_bn
from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.fileio import dump
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmengine.model import MMDistributedDataParallel

from mmengine.runner import set_random_seed
from mmdet3d.registry import DATASETS, MODELS
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

def replace_ImageToTensor(pipelines):
    """Compat helper for deprecated mmdet.datasets.replace_ImageToTensor."""
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug3D':
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" is deprecated for batch inference. '
                'Replacing it with "DefaultFormatBundle".')
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='Whether to fuse conv and bn for speedup')
    parser.add_argument('--format-only', action='store_true', help='Format output results without evaluation')
    parser.add_argument('--eval', type=str, nargs='+', help='evaluation metrics')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where results will be saved')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting results from multiple workers')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config')
    parser.add_argument('--options', nargs='+', action=DictAction, help='deprecated, use --eval-options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError('--options and --eval-options cannot be both specified')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
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

def build_model_compat(model_cfg):
    return MODELS.build(model_cfg)


def build_dataset_compat(dataset_cfg):
    return DATASETS.build(dataset_cfg)


def prepare_test_data_cfg(cfg):
    """Support both legacy cfg.data.test and MMEngine test_dataloader."""
    if 'test_dataloader' in cfg:
        test_loader_cfg = cfg.test_dataloader
        dataset_cfg = test_loader_cfg.dataset
        dataset_cfg.test_mode = True
        samples_per_gpu = test_loader_cfg.get('batch_size', 1)
        workers_per_gpu = test_loader_cfg.get(
            'num_workers', cfg.get('data', {}).get('workers_per_gpu', 2))
        return dataset_cfg, samples_per_gpu, workers_per_gpu

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        return cfg.data.test, samples_per_gpu, cfg.data.workers_per_gpu

    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
    if samples_per_gpu > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    return cfg.data.test, samples_per_gpu, cfg.data.workers_per_gpu


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
        'Please specify at least one operation via --out/--eval/--format-only/--show/--show-dir'

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
    import_plugin_modules(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    test_data_cfg, samples_per_gpu, workers_per_gpu = prepare_test_data_cfg(cfg)
    dataset = build_dataset_compat(test_data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.get('data', {}).get('nonshuffler_sampler', None),
    )

    cfg.model.train_cfg = None
    model = build_model_compat(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    if not distributed:
        raise NotImplementedError('Single GPU testing path is not maintained for this project.')

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            dump(outputs, args.out)

        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join(
            'test',
            args.config.split('/')[-1].split('.')[-2],
            time.ctime().replace(' ', '_').replace(':', '_'))

        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
