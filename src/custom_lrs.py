from torch.optim.lr_scheduler import _LRScheduler

class WarmUpCosineLRScheduler(_LRScheduler):

    def __init__(self, optimizer, after_scheduler, lr_initial, lr_final, warmup_epochs, warmup_method="linear", last_epoch=-1):

        self.warmup_epochs = warmup_epochs
        self.c = lr_initial
        self.a = (lr_final - self.c) / (warmup_epochs ** 2)
        self.b = (lr_final - self.a * warmup_epochs ** 2 - self.c) / warmup_epochs
        self.after_scheduler = after_scheduler

        self.intercept = lr_initial
        self.slope = (lr_final - lr_initial) / warmup_epochs

        self.warmup_method = warmup_method

        super(WarmUpCosineLRScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch <= self.warmup_epochs:
            if self.warmup_method == "linear":
                lr = self.slope * epoch + self.intercept
            elif self.warmup_method == "polynomial":
                lr = self.a * (epoch ** 2) + self.b * epoch + self.c
            return [max(lr, 0) for _ in self.optimizer.param_groups]
        else:
            return self.after_scheduler.get_last_lr()

    def step(self):
        if self.last_epoch <= self.warmup_epochs:
            super(WarmUpCosineLRScheduler, self).step()
        if self.last_epoch > self.warmup_epochs:
            self.after_scheduler.step()


class WarmUpLinearLRScheduler(_LRScheduler):

    def __init__(self, optimizer, after_scheduler, lr_initial, lr_final, warmup_steps, last_epoch=-1):

        self.warmup_steps = warmup_steps
        self.c = lr_initial
        self.m = (lr_final - lr_initial) / warmup_steps
        self.after_scheduler = after_scheduler
        super(WarmUpLinearLRScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_steps:
            lr = self.m * epoch + self.c
            return [max(lr, 0) for _ in self.optimizer.param_groups]
        else:
            return self.after_scheduler.get_last_lr()

    def step(self, bi=None):
        if self.last_epoch < self.warmup_steps:
            super(WarmUpLinearLRScheduler, self).step()
        else:
            self.after_scheduler.step(bi)