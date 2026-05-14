from typing import Dict, Optional, Sequence
import os
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from numpy import random
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample
from mmdet.registry import TRANSFORMS as MMDET_TRANSFORMS
from mmengine.structures import InstanceData

@MMDET_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class PadMultiViewImage(BaseTransform):
    """Pad multi-view images.

    Added keys are ``pad_shape``, ``pad_fixed_size`` and ``pad_size_divisor``.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results: Dict) -> None:
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results['img']
            ]
        else:
            padded_img = [
                mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val)
                for img in results['img']
            ]

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def transform(self, results: Dict) -> Dict:
        self._pad_img(results)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(size={self.size}, '
                f'size_divisor={self.size_divisor}, pad_val={self.pad_val})')

@MMDET_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class NormalizeMultiviewImage(BaseTransform):
    """Normalize multi-view images and append ``img_norm_cfg``."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results: Dict) -> Dict:
        results['img'] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results['img']
        ]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mean={self.mean}, '
                f'std={self.std}, to_rgb={self.to_rgb})')

@MMDET_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage(BaseTransform):
    """Apply photometric distortion to each view image."""

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def transform(self, results: Dict) -> Dict:
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                'PhotoMetricDistortion needs float32 inputs, '
                'please set to_float32=True in image loader')

            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            mode = random.randint(2)
            if mode == 1 and random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha

            img = mmcv.bgr2hsv(img)

            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper)

            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            img = mmcv.hsv2bgr(img)

            if mode == 0 and random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha

            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)

        results['img'] = new_imgs
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'brightness_delta={self.brightness_delta},\n'
                f'contrast_range={(self.contrast_lower, self.contrast_upper)},\n'
                f'saturation_range={(self.saturation_lower, self.saturation_upper)},\n'
                f'hue_delta={self.hue_delta})')

@MMDET_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class CustomCollect3D(BaseTransform):
    """Pack results into MMEngine data flow format.

    Output format:
    - ``inputs``: dict of model input tensors/data
    - ``data_samples``: :class:`Det3DDataSample` with metainfo and gt fields
    """

    def __init__(self,
                 keys: Sequence[str],
                 meta_keys: Sequence[str] = (
                     'filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam',
                     'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                     'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                     'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                     'prev_idx', 'next_idx', 'pcd_scale_factor', 'pcd_rotation',
                     'pts_filename', 'transformation_3d_flow', 'scene_token', 'can_bus')):
        self.keys = keys
        self.meta_keys = meta_keys

    def _is_input_key(self, key: str) -> bool:
        return key in {'img', 'points', 'voxels', 'coors', 'voxel_centers', 'num_points'}

    def transform(self, results: Dict) -> Dict:
        metainfo = {k: results[k] for k in self.meta_keys if k in results}

        data_sample = results.get('data_samples', Det3DDataSample())
        data_sample.set_metainfo(metainfo)

        inputs = results.get('inputs', {})
        if not isinstance(inputs, dict):
            inputs = {'img': inputs}

        for key in self.keys:
            if key not in results:
                continue
            value = results[key]
            if self._is_input_key(key):
                inputs[key] = value
            else:
                if hasattr(data_sample, 'set_field'):
                    data_sample.set_field(value, key)
                else:
                    setattr(data_sample, key, value)

        if hasattr(data_sample, 'gt_bboxes_3d') or hasattr(data_sample, 'gt_labels_3d'):
            gt_instances_3d = getattr(data_sample, 'gt_instances_3d', InstanceData())
            if hasattr(data_sample, 'gt_bboxes_3d'):
                gt_instances_3d.bboxes_3d = data_sample.gt_bboxes_3d
            if hasattr(data_sample, 'gt_labels_3d'):
                gt_instances_3d.labels_3d = data_sample.gt_labels_3d
            if hasattr(data_sample, 'set_field'):
                data_sample.set_field(gt_instances_3d, 'gt_instances_3d')
            else:
                data_sample.gt_instances_3d = gt_instances_3d

        if os.getenv('DEBUG_COLLECT3D_GT_LABELS', '0') == '1':
            gt = getattr(data_sample, 'gt_labels_3d', None)
            if gt is None and hasattr(data_sample, 'get'):
                gt = data_sample.get('gt_labels_3d', None)
            print('[CustomCollect3D] sample_idx=', metainfo.get('sample_idx', None),
                  'gt_labels_3d=', gt)

        if os.getenv('DEBUG_COLLECT3D_GT_LABELS', '0') == '1':
            gt = getattr(data_sample, 'gt_labels_3d', None)
            if gt is None and hasattr(data_sample, 'get'):
                gt = data_sample.get('gt_labels_3d', None)
            print('[CustomCollect3D] sample_idx=', metainfo.get('sample_idx', None),
                  'gt_labels_3d=', gt)

        packed = dict(inputs=inputs, data_samples=data_sample)
        return packed

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys}, meta_keys={self.meta_keys})'

@MMDET_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class RandomScaleImageMultiViewImage(BaseTransform):
    """Random scale multi-view images (single scale currently)."""

    def __init__(self, scales: Optional[Sequence[float]] = None):
        self.scales = list(scales or [])
        assert len(self.scales) == 1

    def transform(self, results: Dict) -> Dict:
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale

        results['img'] = [
            mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False)
            for idx, img in enumerate(results['img'])
        ]
        results['lidar2img'] = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(size={self.scales})'
