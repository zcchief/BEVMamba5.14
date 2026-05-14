from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class TransferWeight(Hook):

    def __init__(self, every_n_iters=1, every_n_inters=None):
        # keep backward compatibility with old typo argument: every_n_inters
        if every_n_inters is not None:
            every_n_iters = every_n_inters
        self.every_n_iters = every_n_iters

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if not self.every_n_train_iters(runner, self.every_n_iters):
            return
        if not hasattr(runner, 'eval_model') or runner.eval_model is None:
            return

        src_model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        dst_model = runner.eval_model.module if hasattr(runner.eval_model, 'module') else runner.eval_model
        dst_model.load_state_dict(src_model.state_dict())

