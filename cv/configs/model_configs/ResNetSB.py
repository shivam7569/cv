def ResNetSBConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ResNetSB = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ResNetSB"
    cfg.METRICS.NAME = "ResNetSB"
    cfg.CHECKPOINT.BASENAME = "ResNetSB"
    cfg.TENSORBOARD.BASENAME = "ResNetSB"
    cfg.PROFILER.BASENAME = "ResNetSB/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.ResNetSB.PARAMS = CN()
    cfg.ResNetSB.PARAMS.num_classes = 1000
    cfg.ResNetSB.PARAMS.in_channels = 3
    cfg.ResNetSB.PARAMS.layer_scale = None
    cfg.ResNetSB.PARAMS.stochastic_depth_mp = 0.05

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 700
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 2048
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None

    cfg.DATA_MIXING.enabled = True
    cfg.DATA_MIXING.mixup_alpha = 0.2
    cfg.DATA_MIXING.cutmix_alpha = 1.0
    cfg.DATA_MIXING.cutmix_minmax = None
    cfg.DATA_MIXING.prob = 0.5
    cfg.DATA_MIXING.switch_prob = 0.5
    cfg.DATA_MIXING.mode = "batch"
    cfg.DATA_MIXING.correct_lam = True
    cfg.DATA_MIXING.label_smoothing = 0.0
    cfg.DATA_MIXING.one_hot_targets = False
    cfg.DATA_MIXING.num_classes = 1000

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.ResNetSB.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ResNetSB.DATALOADER_TRAIN_PARAMS.batch_size = 96
    cfg.ResNetSB.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ResNetSB.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.ResNetSB.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ResNetSB.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ResNetSB.DATALOADER_VAL_PARAMS = CN()
    cfg.ResNetSB.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.ResNetSB.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ResNetSB.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ResNetSB.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ResNetSB.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ResNetSB.TRANSFORMS = CN()

    cfg.ResNetSB.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=7, n=3, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    ]

    cfg.ResNetSB.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ResNetSB.OPTIMIZER = CN()
    cfg.ResNetSB.OPTIMIZER.NAME = "Lamb"
    cfg.ResNetSB.OPTIMIZER.PARAMS = CN()
    cfg.ResNetSB.OPTIMIZER.PARAMS.lr = 5e-3 * cfg.num_gpus
    cfg.ResNetSB.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.ResNetSB.OPTIMIZER.PARAMS.weight_decay = 0.01

    cfg.ResNetSB.LR_SCHEDULER = CN()
    cfg.ResNetSB.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.ResNetSB.LR_SCHEDULER.PARAMS = CN()
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.lr_final = cfg.ResNetSB.OPTIMIZER.PARAMS.lr
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.ResNetSB.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1

    cfg.ResNetSB.LOSS = CN()
    cfg.ResNetSB.LOSS.NAME = "CrossEntropyLoss"
    cfg.ResNetSB.LOSS.PARAMS = CN()
    cfg.ResNetSB.LOSS.PARAMS.weight = None
    cfg.ResNetSB.LOSS.PARAMS.size_average = None
    cfg.ResNetSB.LOSS.PARAMS.ignore_index = -100
    cfg.ResNetSB.LOSS.PARAMS.reduce = None
    cfg.ResNetSB.LOSS.PARAMS.reduction = "mean"
    cfg.ResNetSB.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0