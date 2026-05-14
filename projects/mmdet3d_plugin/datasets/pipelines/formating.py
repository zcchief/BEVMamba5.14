# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample



@TRANSFORMS.register_module()
class CustomPackDet3DInputs(BaseTransform):
    """自定义数据打包 Transform，兼容新版 MMEngine 数据流。

    该 Transform 将 pipeline 处理后的中间结果字典转换为模型输入的标准格式：
        {
            'inputs': Union[torch.Tensor, Dict[str, torch.Tensor]],
            'data_samples': Det3DDataSample
        }

    同时处理自定义字段 `gt_map_masks`，将其转换为 tensor 并存入 data_samples。
    """

    def __init__(self,
                 keys: tuple = ('img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'),
                 meta_keys: tuple = ('img_metas',)):
        self.keys = keys
        self.meta_keys = meta_keys

    def transform(self, results: Dict) -> Dict:
        """转换数据格式。

        Args:
            results (dict): Pipeline 输出的中间结果字典。

        Returns:
            dict: 包含 'inputs' 和 'data_samples' 的标准数据包。
        """
        data_sample = Det3DDataSample()

        if 'gt_instances_3d' in results:
            data_sample.gt_instances_3d = results['gt_instances_3d']
        if 'ignored_instances' in results:
            data_sample.ignored_instances = results['ignored_instances']

        metainfo = {}
        for key in self.meta_keys:
            if key in results:
                metainfo[key] = results[key]
        if 'img_path' in results:
            metainfo['img_path'] = results['img_path']
        if 'lidar2img' in results:
            metainfo['lidar2img'] = results['lidar2img']
        if 'scene_token' in results:
            metainfo['scene_token'] = results['scene_token']
        data_sample.set_metainfo(metainfo)

        if 'gt_map_masks' in results:
            gt_map_masks = results['gt_map_masks']
            if isinstance(gt_map_masks, np.ndarray):
                gt_map_masks = torch.from_numpy(gt_map_masks).float()
            elif isinstance(gt_map_masks, torch.Tensor):
                gt_map_masks = gt_map_masks.float()
            else:
                raise TypeError(
                    f'Unsupported type {type(gt_map_masks)} for gt_map_masks')
            data_sample.set_field(gt_map_masks, 'gt_map_masks')

        inputs = {}
        for key in self.keys:
            if key not in results:
                continue
            value = results[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif isinstance(value, (list, tuple)):
                value = torch.stack([to_tensor(img) for img in value])
            elif isinstance(value, torch.Tensor):
                pass
            else:
                if hasattr(value, 'tensor'):
                    value = value.tensor
                else:
                    continue
            inputs[key] = value

        if len(inputs) == 1:
            inputs = list(inputs.values())[0]

        return dict(inputs=inputs, data_samples=data_sample)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')
