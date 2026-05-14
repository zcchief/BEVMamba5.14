# Copyright (c) OpenMMLab. All rights reserved.
import platform
import random
from functools import partial
from collections import defaultdict

import numpy as np
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.dist import get_dist_info
from mmengine.registry import Registry
from torch.utils.data import BatchSampler, DataLoader

from mmengine.registry import DATASETS
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import DistributedGroupSampler
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import DistributedSampler
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler



def _concat_dataset_compat(cfg, default_args=None):
    """Build concat dataset when ann_file is a list/tuple (MMDet3-compatible)."""
    from mmengine.dataset import ConcatDataset

    ann_files = cfg.get('ann_file')
    assert isinstance(ann_files, (list, tuple))

    datasets = []
    for i, ann_file in enumerate(ann_files):
        data_cfg = cfg.copy()
        data_cfg['ann_file'] = ann_file

        for key in ['img_prefix', 'seg_prefix', 'proposal_file']:
            if key in data_cfg and isinstance(data_cfg[key], (list, tuple)):
                data_cfg[key] = data_cfg[key][i]

        datasets.append(DATASETS.build(data_cfg, default_args=default_args))

    return ConcatDataset(datasets)


class AspectRatioBatchSampler(BatchSampler):
    """A lightweight aspect-ratio batch sampler compatible with MMDet3.x.

    This sampler groups indices by ``dataset.flag`` first, then forms batches
    within each group to mimic the old GroupSampler behavior.
    """

    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.flag = getattr(sampler.dataset, 'flag', None)

    def __iter__(self):
        if self.flag is None:
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
            return

        buckets = defaultdict(list)
        for idx in self.sampler:
            group = int(self.flag[idx])
            buckets[group].append(idx)
            if len(buckets[group]) == self.batch_size:
                yield buckets[group]
                buckets[group] = []

        if not self.drop_last:
            for group in buckets:
                if len(buckets[group]) > 0:
                    yield buckets[group]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    batch_sampler = None
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = DefaultSampler(dataset, shuffle=shuffle, seed=seed)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            batch_size=samples_per_gpu,
            drop_last=False)
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    loader_kwargs = dict(
        dataset=dataset,
        num_workers=num_workers,
        collate_fn=pseudo_collate,
        pin_memory=False,
        worker_init_fn=init_fn,
        persistent_workers=(num_workers > 0),
        **kwargs)
    if batch_sampler is not None:
        loader_kwargs['batch_sampler'] = batch_sampler
    else:
        loader_kwargs['batch_size'] = batch_size
        loader_kwargs['sampler'] = sampler

    data_loader = DataLoader(**loader_kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def custom_build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmengine.dataset import (ClassBalancedDataset, ConcatDataset, RepeatDataset)

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([custom_build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(custom_build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset_compat(cfg, default_args)
    else:
        dataset = DATASETS.build(cfg, default_args=default_args)

    return dataset