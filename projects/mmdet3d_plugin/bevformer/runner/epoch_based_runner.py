# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmengine.registry import LOOPS, RUNNERS
from mmengine.runner.loops import EpochBasedTrainLoop


@LOOPS.register_module(name='EpochBasedTrainLoop_video')
@LOOPS.register_module(name='EpochBasedTrainLoopVideo')
class EpochBasedTrainLoopVideo(EpochBasedTrainLoop):
    """OpenMMLab 2.0 loop version of legacy EpochBasedRunner_video."""

    def __init__(self,
                 runner,
                 dataloader,
                 max_epochs,
                 val_begin=1,
                 val_interval=1,
                 dynamic_intervals=None,
                 keys=('gt_bboxes_3d', 'gt_labels_3d', 'img')):
        super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)
        self.keys = list(keys)
        if 'img_metas' not in self.keys:
            self.keys.append('img_metas')

    @property
    def eval_model(self):
        return getattr(self.runner, 'eval_model', None)

    @staticmethod
    def _unwrap(item):
        if hasattr(item, 'data'):
            data = item.data
            return data[0] if isinstance(data, list) else data
        if isinstance(item, list) and len(item) == 1:
            return item[0]
        return item

    def _build_frame_batches(self, data_batch):
        queue_metas = None
        if 'img' in data_batch and 'img_metas' in data_batch:
            imgs = self._unwrap(data_batch['img'])
            img_metas = self._unwrap(data_batch['img_metas'])
        else:
            inputs = data_batch.get('inputs', {})
            imgs = self._unwrap(inputs['img'] if isinstance(inputs, dict) else inputs)
            data_samples = data_batch.get('data_samples', None)
            if isinstance(data_samples, list):
                img_metas = [ds.metainfo if hasattr(ds, 'metainfo') else ds for ds in data_samples]
                if len(data_samples) == 1 and hasattr(data_samples[0], 'metainfo'):
                    queue_metas = data_samples[0].metainfo.get('queue_metas', None)
            elif data_samples is not None and hasattr(data_samples, 'metainfo'):
                img_metas = data_samples.metainfo
                queue_metas = data_samples.metainfo.get('queue_metas', None)
            else:
                img_metas = data_samples

        if queue_metas is not None:
            num_samples = len(queue_metas)
        elif isinstance(img_metas, dict):
            num_samples = len(img_metas)
        else:
            num_samples = imgs.size(1)

        data_list = []
        for i in range(num_samples):
            data = {}
            for key in self.keys:
                if key not in ['img_metas', 'img', 'points']:
                    if key in data_batch:
                        data[key] = data_batch[key]
                elif key == 'img':
                    if imgs.ndim >= 5 and imgs.size(0) == num_samples:
                        data['img'] = imgs[i]
                    else:
                        data['img'] = imgs[:, i]
                elif key == 'img_metas':
                    if queue_metas is not None:
                        data['img_metas'] = [queue_metas[i]]
                    elif isinstance(img_metas, dict):
                        data['img_metas'] = [img_metas[i]]
                    else:
                        data['img_metas'] = [each[i] for each in img_metas]
            data_list.append(data)
        return data_list

    def run_iter(self, idx, data_batch):
        eval_model = self.eval_model
        if eval_model is None:
            # Fallback to default loop when temporal frozen branch is absent.
            return super().run_iter(idx, data_batch)

        eval_model.eval()
        data_list = self._build_frame_batches(data_batch)
        prev_bev = None
        with torch.no_grad():
            for i in range(len(data_list) - 1):
                if data_list[i]['img_metas'][0]['prev_bev_exists']:
                    data_list[i]['prev_bev'] = prev_bev
                try:
                    prev_bev = eval_model.val_step(data_list[i])
                except TypeError:
                    prev_bev = eval_model.val_step(data_list[i], self.runner.optim_wrapper)

        if data_list[-1]['img_metas'][0]['prev_bev_exists']:
            data_list[-1]['prev_bev'] = prev_bev

        return super().run_iter(idx, data_list[-1])


@RUNNERS.register_module()
class EpochBasedRunner_video:  # pragma: no cover - compatibility shim
    """Legacy runner name is no longer supported in OpenMMLab 2.0."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'EpochBasedRunner_video has been migrated to Loop mode. '
            'Please use `train_cfg=dict(type=\"EpochBasedTrainLoopVideo\", ...)` '
            'with mmengine.Runner.'
        )