
import bisect
import os.path as osp

import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.utils import is_list_of
from torch.nn.modules.batchnorm import _BatchNorm


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(Hook):

    def __init__(self,
                 dataloader,
                 interval=1,
                 by_epoch=True,
                 tmpdir=None,
                 gpu_collect=False,
                 save_best=None,
                 broadcast_bn_buffer=True,
                 test_fn=None,
                 dynamic_intervals=None,
                 **eval_kwargs):
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect
        self.save_best = save_best
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.test_fn = test_fn
        self.eval_kwargs = eval_kwargs

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = _calc_dynamic_intervals(
                self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            self.interval = self.dynamic_intervals[step - 1]

    def _should_evaluate(self, runner):
        if self.by_epoch:
            return self.every_n_epochs(runner, self.interval)
        return self.every_n_train_iters(runner, self.interval)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self._decide_interval(runner)
        self._do_evaluate(runner)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if self.by_epoch:
            return
        self._decide_interval(runner)
        if self.every_n_train_iters(runner, self.interval):
            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """Perform evaluation and save ckpt."""
        if self.broadcast_bn_buffer and dist.is_available() and dist.is_initialized():
            model = runner.model
            for _, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # solve circular import
        from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test

        test_fn = self.test_fn or custom_multi_gpu_test
        results = test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)

        rank = getattr(runner, 'rank', 0)
        if rank == 0:
            print('\n')
            if hasattr(runner, 'log_buffer'):
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)
            if self.save_best:
                runner.logger.info('`save_best` is set, but checkpoint comparison '
                                   'is not implemented in CustomDistEvalHook.')
            return key_score

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        rank = getattr(runner, 'rank', 0)
        if rank == 0 and hasattr(runner, 'log_buffer'):
            runner.log_buffer.output.update(eval_res)
            runner.log_buffer.ready = True

        return eval_res
