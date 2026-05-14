# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import copy

import torch
from mmengine.dist import get_dist_info
from mmengine.model import MMDistributedDataParallel
from mmengine.registry import HOOKS, build_from_cfg
from mmengine.runner import Runner

from mmengine.logging import MMLogger
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def _convert_lr_config_to_param_scheduler(lr_config, max_epochs):
    """Best-effort conversion from legacy lr_config to MMEngine scheduler."""
    if not lr_config:
        return None

    schedulers = []
    warmup = lr_config.get('warmup', None)
    if warmup == 'linear':
        schedulers.append(
            dict(
                type='LinearLR',
                start_factor=lr_config.get('warmup_ratio', 0.1),
                begin=0,
                end=lr_config.get('warmup_iters', 0),
                by_epoch=False))

    policy = lr_config.get('policy', None)
    if policy == 'CosineAnnealing':
        schedulers.append(
            dict(
                type='CosineAnnealingLR',
                T_max=max_epochs,
                eta_min=lr_config.get('min_lr', 0),
                begin=0,
                end=max_epochs,
                by_epoch=True))

    return schedulers if schedulers else None


def _infer_by_epoch(cfg):
    """Infer eval rhythm from loop config first, then legacy runner type."""
    train_cfg = cfg.get('train_cfg', None)
    if isinstance(train_cfg, dict) and 'type' in train_cfg:
        return 'IterBased' not in train_cfg['type']
    return cfg.get('runner', {}).get('type', 'EpochBasedRunner') != 'IterBasedRunner'


def _convert_legacy_default_hooks(cfg):
    """Merge legacy log/checkpoint configs into MMEngine default_hooks."""
    default_hooks = copy.deepcopy(cfg.get('default_hooks', {}))

    checkpoint_cfg = cfg.get('checkpoint_config', None)
    if checkpoint_cfg is not None:
        ckpt_hook = copy.deepcopy(default_hooks.get('checkpoint', {}))
        ckpt_hook.update(copy.deepcopy(checkpoint_cfg))
        default_hooks['checkpoint'] = ckpt_hook

    log_cfg = cfg.get('log_config', None)
    if log_cfg is not None:
        logger_hook = copy.deepcopy(default_hooks.get('logger', {}))
        if 'interval' in log_cfg:
            logger_hook.setdefault('interval', log_cfg['interval'])
        if isinstance(log_cfg.get('hooks', None), list) and log_cfg['hooks']:
            first_hook = log_cfg['hooks'][0]
            if isinstance(first_hook, dict) and 'interval' in first_hook:
                logger_hook.setdefault('interval', first_hook['interval'])
        default_hooks['logger'] = logger_hook

    return default_hooks if default_hooks else None


def _to_runner_cfg(cfg, train_dataloader, val_dataloader=None, validate=False):
    """Convert legacy cfg fragments to MMEngine Runner style config."""
    runner_cfg = dict()

    # loop cfg
    if 'train_cfg' in cfg:
        runner_cfg['train_cfg'] = copy.deepcopy(cfg.train_cfg)
    else:
        legacy_runner_type = cfg.get('runner', {}).get('type', 'EpochBasedRunner')
        if legacy_runner_type == 'IterBasedRunner':
            max_iters = cfg.get('runner', {}).get('max_iters', cfg.get('total_iters', 1))
            runner_cfg['train_cfg'] = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1)
        else:
            max_epochs = cfg.get('total_epochs', cfg.get('runner', {}).get('max_epochs', 1))
            runner_cfg['train_cfg'] = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

    #if validate:
        #runner_cfg['val_cfg'] = copy.deepcopy(cfg.get('val_cfg', dict(type='ValLoop')))
    # only set test loop when complete test triplet is provided
    if ('test_dataloader' in cfg) and ('test_evaluator' in cfg):
        runner_cfg['test_cfg'] = copy.deepcopy(cfg.get('test_cfg', dict(type='TestLoop')))
        runner_cfg['test_dataloader'] = copy.deepcopy(cfg.test_dataloader)
        runner_cfg['test_evaluator'] = copy.deepcopy(cfg.test_evaluator)

    # dataloader
    runner_cfg['train_dataloader'] = train_dataloader
    #if validate and val_dataloader is not None:
        #runner_cfg['val_dataloader'] = val_dataloader

    # optimizer migration: legacy optimizer -> optim_wrapper
    if 'optim_wrapper' in cfg:
        runner_cfg['optim_wrapper'] = copy.deepcopy(cfg.optim_wrapper)
    else:
        if 'optimizer' not in cfg:
            raise KeyError('Either cfg.optim_wrapper or cfg.optimizer must be provided for MMEngine runner.')
        runner_cfg['optim_wrapper'] = dict(type='OptimWrapper', optimizer=copy.deepcopy(cfg.optimizer))

    optimizer_config = cfg.get('optimizer_config', None)
    if isinstance(optimizer_config, dict) and optimizer_config.get('grad_clip', None) is not None:
        runner_cfg['optim_wrapper']['clip_grad'] = copy.deepcopy(optimizer_config['grad_clip'])

    # scheduler migration
    if 'param_scheduler' in cfg:
        runner_cfg['param_scheduler'] = copy.deepcopy(cfg.param_scheduler)
    elif 'lr_config' in cfg:
        max_epochs = runner_cfg['train_cfg'].get('max_epochs', cfg.get('total_epochs', 1))
        schedulers = _convert_lr_config_to_param_scheduler(cfg.lr_config, max_epochs)
        if schedulers is not None:
            runner_cfg['param_scheduler'] = schedulers

    default_hooks = _convert_legacy_default_hooks(cfg)
    if default_hooks is not None:
        runner_cfg['default_hooks'] = default_hooks

    passthrough_keys = [
        'env_cfg', 'launcher', 'default_scope', 'randomness', 'log_level',
        'log_processor', 'visualizer', 'experiment_name',
    ]
    for key in passthrough_keys:
        if key in cfg:
            runner_cfg[key] = copy.deepcopy(cfg[key])

    if 'load_from' in cfg:
        runner_cfg['load_from'] = cfg.load_from
    if cfg.get('resume_from', None):
        runner_cfg['resume'] = True
        runner_cfg['load_from'] = cfg.resume_from

    return runner_cfg


def custom_train_detector(model,
                          dataset,
                          cfg,
                          distributed=False,
                          validate=False,
                          timestamp=None,
                          eval_model=None,
                          meta=None):
    """MMEngine-runner based custom training entry for BEVFormer plugin."""
    logger = MMLogger.get_instance('mmdet', log_level=cfg.log_level)

    datasets = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    train_dataset = datasets[0]
    if not hasattr(cfg, 'gpu_ids') or cfg.gpu_ids is None:
        if distributed:
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)
        else:
            cfg.gpu_ids = range(max(torch.cuda.device_count(), 1))

    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated. Please use "samples_per_gpu".')
        cfg.data.samples_per_gpu = cfg.data.get('samples_per_gpu', cfg.data.imgs_per_gpu)

    train_dataloader = build_dataloader(
        train_dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    val_dataloader = None
    eval_cfg = None
    if validate:
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler)
        eval_cfg = cfg.get('evaluation', {}).copy()
        eval_cfg['by_epoch'] = _infer_by_epoch(cfg)

    if eval_model is None and cfg.get('runner', {}).get('type', None) == 'EpochBasedRunner_video':
        # Keep BEVFormer temporal training flow: a frozen eval branch is used
        # to extract prev_bev from previous frames.
        eval_model = copy.deepcopy(model)

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if len(cfg.gpu_ids) == 1:
            model = model.cuda(cfg.gpu_ids[0])
            if eval_model is not None:
                eval_model = eval_model.cuda(cfg.gpu_ids[0])
        else:
            model = torch.nn.DataParallel(model.cuda(), device_ids=cfg.gpu_ids)
            if eval_model is not None:
                eval_model = torch.nn.DataParallel(eval_model.cuda(), device_ids=cfg.gpu_ids)

    runner_cfg = _to_runner_cfg(cfg, train_dataloader=None, val_dataloader=None, validate=False)

    # 取出可直接传给 Runner(...) 的“纯配置字段”
    train_cfg = runner_cfg.get('train_cfg', None)
    optim_wrapper = runner_cfg.get('optim_wrapper', None)
    param_scheduler = runner_cfg.get('param_scheduler', None)
    default_hooks = runner_cfg.get('default_hooks', None) or {}
    env_cfg = runner_cfg.get('env_cfg', None) or {}
    launcher = runner_cfg.get('launcher', 'none')
    randomness = runner_cfg.get('randomness', None)
    if randomness is None:
        randomness = dict(
            seed=cfg.get('seed', 0),
            deterministic=getattr(cfg, 'deterministic', False)
        )
    log_level = runner_cfg.get('log_level', cfg.get('log_level', 'INFO'))
    log_processor = runner_cfg.get('log_processor', None) or dict(type='LogProcessor', window_size=50, by_epoch=True)
    visualizer = runner_cfg.get('visualizer', None)
    load_from = runner_cfg.get('load_from', None)
    resume = runner_cfg.get('resume', False)



    experiment_name = None
    if meta is not None and meta.get('exp_name', None) is not None:
        experiment_name = meta.get('exp_name')
    else:
        experiment_name = runner_cfg.get('experiment_name', None)

    runner = Runner(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        default_hooks=default_hooks,
        env_cfg=env_cfg,
        launcher=launcher,
        randomness=randomness,
        log_level=log_level,
        log_processor=log_processor,
        visualizer=visualizer,
        load_from=load_from,
        resume=resume,
        experiment_name=experiment_name
    )

    # keep backward compatibility for custom temporal runner hooks
    runner.eval_model = eval_model

    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in custom_hooks:
            assert isinstance(hook_cfg, dict), f'Each custom_hook should be a dict, got {type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if validate and val_dataloader is not None and eval_cfg is not None:
        eval_hook = CustomDistEvalHook(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook)

    runner.train()