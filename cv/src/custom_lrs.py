from torch.optim.lr_scheduler import _LRScheduler

from cv.utils import MetaWrapper

class WarmUpCosineLRScheduler(_LRScheduler, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Cosine Learning Rate Scheduler with Linear Warmup"

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


class WarmUpLinearLRScheduler(_LRScheduler, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "User Defined Learning Rate Scheduler with Linear Warmup"

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

class PolynomialLRScheduler(_LRScheduler, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Polynomial Learning Rate Scheduler"
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError("max_decay_steps should be greater than 1")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group["lr"] = lr

class WarmUpPolyLRScheduler(_LRScheduler, metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Polynomial Learning Rate Scheduler with Linear Warmup"

    def __init__(self, optimizer, after_scheduler, lr_initial, lr_final, warmup_epochs, warmup_method="linear", last_epoch=-1):

        self.warmup_epochs = warmup_epochs
        self.c = lr_initial
        self.a = (lr_final - self.c) / (warmup_epochs ** 2)
        self.b = (lr_final - self.a * warmup_epochs ** 2 - self.c) / warmup_epochs
        self.after_scheduler = after_scheduler

        self.intercept = lr_initial
        self.slope = (lr_final - lr_initial) / warmup_epochs

        self.warmup_method = warmup_method

        super(WarmUpPolyLRScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch <= self.warmup_epochs:
            if self.warmup_method == "linear":
                lr = self.slope * epoch + self.intercept
            elif self.warmup_method == "polynomial":
                lr = self.a * (epoch ** 2) + self.b * epoch + self.c
            return [max(lr, 0) for _ in self.optimizer.param_groups]
        else:
            return self.after_scheduler.get_lr()

    def step(self):
        if self.last_epoch <= self.warmup_epochs:
            super(WarmUpPolyLRScheduler, self).step()
        if self.last_epoch >= self.warmup_epochs:
            self.after_scheduler.step()