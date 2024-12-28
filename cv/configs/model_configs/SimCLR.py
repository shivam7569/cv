import random

def SimCLRConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.SimCLR = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAME_CONTRASTIVE_TRANSFORMS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "SimCLR"
    cfg.METRICS.NAME = "SimCLR"
    cfg.CHECKPOINT.BASENAME = "SimCLR"
    cfg.TENSORBOARD.BASENAME = "SimCLR"
    cfg.PROFILER.BASENAME = "SimCLR/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.SimCLR.PARAMS = CN()
    cfg.SimCLR.PARAMS.backbone = "ResNet"
    cfg.SimCLR.PARAMS.backbone_params = CN()
    cfg.SimCLR.PARAMS.backbone_params.num_classes = 1000
    cfg.SimCLR.PARAMS.projection_dim = 128

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 1000
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 4096
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.SimCLR.DATALOADER_TRAIN_PARAMS = CN()
    cfg.SimCLR.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.SimCLR.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.SimCLR.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.SimCLR.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.SimCLR.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.SimCLR.DATALOADER_VAL_PARAMS = CN()
    cfg.SimCLR.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.SimCLR.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.SimCLR.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.SimCLR.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.SimCLR.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.SimCLR.TRANSFORMS = CN()

    cfg.SimCLR.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2))
            ], p=0.5
        )),
        dict(name="RandomGrayscale", params=dict(p=0.2)),
        dict(name="GaussianBlur", params=dict(kernel_size=random.choice(range(20, 25)))),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="RandomEqualize", params=dict(p=1.0)),
                dict(name="RandomSolarize", params=dict(threshold=200, p=1.0)),
            ], p=0.2
        )),
        dict(name="SobelFilter", params=dict(p=0.2)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.SimCLR.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.SimCLR.OPTIMIZER = CN()
    cfg.SimCLR.OPTIMIZER.NAME = "Adam"
    cfg.SimCLR.OPTIMIZER.PARAMS = CN()
    cfg.SimCLR.OPTIMIZER.PARAMS.lr = 0.4
    cfg.SimCLR.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.SimCLR.OPTIMIZER.PARAMS.weight_decay = 1e-6

    cfg.SimCLR.LR_SCHEDULER = CN()
    cfg.SimCLR.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.SimCLR.LR_SCHEDULER.PARAMS = CN()
    cfg.SimCLR.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.SimCLR.LR_SCHEDULER.PARAMS.lr_final = cfg.SimCLR.OPTIMIZER.PARAMS.lr
    cfg.SimCLR.LR_SCHEDULER.PARAMS.warmup_epochs = 10
    cfg.SimCLR.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.SimCLR.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.SimCLR.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingLR"
    cfg.SimCLR.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.SimCLR.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_max = 990
    cfg.SimCLR.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.SimCLR.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.SimCLR.LOSS = CN()
    cfg.SimCLR.LOSS.NAME = "NT_XentLoss"
    cfg.SimCLR.LOSS.PARAMS = CN()
    cfg.SimCLR.LOSS.PARAMS.temperature = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0