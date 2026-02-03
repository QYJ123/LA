from mmdet.engine.hooks import EvalHook
from mmengine.runner import Runner

class CustomEvalHook(EvalHook):
   
    def __init__(self, start_eval_epoch, eval_interval=1, **kwargs):
        super().__init__(interval=eval_interval, **kwargs)
        self.start_eval_epoch = start_eval_epoch  # 开始评估的起始epoch

    def after_train_epoch(self, runner: Runner):
       
        if runner.epoch < self.start_eval_epoch:
            return

        super().after_train_epoch(runner)

# 在你的配置文件中，添加这个自定义Hook
custom_hooks = [
    dict(
        type='CustomEvalHook',
        start_eval_epoch=90,  # 从第90个epoch开始
        eval_interval=5,       # 之后每5个epoch评估一次
        # ... 其他EvalHook的参数，如dataset、metric等
    )
]
