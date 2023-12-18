from torch.optim.lr_scheduler import _LRScheduler

class WarmUpPolynomialLRScheduler(_LRScheduler):

    def __init__(self, optimizer, after_scheduler, lr_initial, lr_final, n_epochs, last_epoch=-1):

        self.n_epochs = n_epochs
        self.c = lr_initial
        self.a = (lr_final - self.c) / (n_epochs ** 2)
        self.b = (lr_final - self.a * n_epochs ** 2 - self.c) / n_epochs
        self.after_scheduler = after_scheduler

        super(WarmUpPolynomialLRScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch <= self.n_epochs:
            lr = self.a * (epoch ** 2) + self.b * epoch + self.c
            return [max(lr, 0) for _ in self.optimizer.param_groups]
        else:
            return self.after_scheduler.get_last_lr()

    def step(self):
        if self.last_epoch <= self.n_epochs:
            super(WarmUpPolynomialLRScheduler, self).step()
        if self.last_epoch > self.n_epochs:
            self.after_scheduler.step()