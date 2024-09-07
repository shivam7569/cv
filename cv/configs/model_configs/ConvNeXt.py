def ConvNeXtConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ConvNeXt = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True
    cfg.REPEAT_AUGMENTATIONS_NUM_REPEATS = 4

    cfg.DEBUG = None
    cfg.USE_SYNC_BN = True
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ConvNeXt"
    cfg.METRICS.NAME = "ConvNeXt"
    cfg.CHECKPOINT.BASENAME = "ConvNeXt"
    cfg.TENSORBOARD.BASENAME = "ConvNeXt"
    cfg.PROFILER.BASENAME = "ConvNeXt/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.ConvNeXt.PARAMS = CN()
    cfg.ConvNeXt.PARAMS.num_classes = 1000
    cfg.ConvNeXt.PARAMS.in_channels = 3
    cfg.ConvNeXt.PARAMS.stem_out_channels = 128
    cfg.ConvNeXt.PARAMS.stem_kernel_size = 4
    cfg.ConvNeXt.PARAMS.stem_kernel_stride = 4
    cfg.ConvNeXt.PARAMS.num_blocks = [3, 3, 27, 3]
    cfg.ConvNeXt.PARAMS.expansion_rate = 4
    cfg.ConvNeXt.PARAMS.depthwise_conv_kernel_size = 7
    cfg.ConvNeXt.PARAMS.layer_scale = 1e-6
    cfg.ConvNeXt.PARAMS.stochastic_depth_mp = 0.1

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 700
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 1024
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = CN()
    cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.999
    cfg.TRAIN.PARAMS.exponential_moving_average.warmup_steps = 0
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_period = 32
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_method = "constant"

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
        dict(func="resizeWithAspectRatio", params=dict(size=232))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=232))
    ]

    cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.drop_last = True
    cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.prefetch_factor = 2

    cfg.ConvNeXt.DATALOADER_VAL_PARAMS = CN()
    cfg.ConvNeXt.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.ConvNeXt.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ConvNeXt.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ConvNeXt.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ConvNeXt.DATALOADER_VAL_PARAMS.drop_last = True
    cfg.ConvNeXt.DATALOADER_VAL_PARAMS.prefetch_factor = 2

    cfg.ConvNeXt.TRANSFORMS = CN()

    cfg.ConvNeXt.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(176, 176), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="TrivialAugmentWide", params=dict()),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.1, mode="pixel", device="cpu")) # to be used after normalization when mode is pixel, as it will avoid changing the image statistics
    ]

    cfg.ConvNeXt.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ConvNeXt.OPTIMIZER = CN()
    cfg.ConvNeXt.OPTIMIZER.NAME = "AdamW"
    cfg.ConvNeXt.OPTIMIZER.PARAMS = CN()
    cfg.ConvNeXt.OPTIMIZER.PARAMS.lr = 1e-3
    cfg.ConvNeXt.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.ConvNeXt.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.ConvNeXt.LR_SCHEDULER = CN()
    cfg.ConvNeXt.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS = CN()
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.lr_initial = 0.0
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.lr_final = cfg.ConvNeXt.OPTIMIZER.PARAMS.lr
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingLR"
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_max = 695
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 0.0
    cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.ConvNeXt.LOSS = CN()
    cfg.ConvNeXt.LOSS.NAME = "CrossEntropyLoss"
    cfg.ConvNeXt.LOSS.PARAMS = CN()
    cfg.ConvNeXt.LOSS.PARAMS.weight = None
    cfg.ConvNeXt.LOSS.PARAMS.size_average = None
    cfg.ConvNeXt.LOSS.PARAMS.ignore_index = -100
    cfg.ConvNeXt.LOSS.PARAMS.reduce = None
    cfg.ConvNeXt.LOSS.PARAMS.label_smoothing = 0.1
    cfg.ConvNeXt.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0


# Original Paper Recipe

# def ConvNeXtConfig(cfg):
#     from cv.configs.config import CfgNode as CN

#     cfg.ConvNeXt = CN()

#     cfg.ASYNC_TRAINING = True
#     cfg.REPEAT_AUGMENTATIONS = True
#     cfg.REPEAT_AUGMENTATIONS_NUM_REPEATS = 4

#     cfg.DEBUG = None
#     cfg.PROFILING = False

#     cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
#     cfg.LOGGING.NAME = "ConvNeXt"
#     cfg.METRICS.NAME = "ConvNeXt"
#     cfg.CHECKPOINT.BASENAME = "ConvNeXt"
#     cfg.TENSORBOARD.BASENAME = "ConvNeXt"
#     cfg.PROFILER.BASENAME = "ConvNeXt/Exp"
#     cfg.CHECKPOINT.TREE = False

#     cfg.ConvNeXt.PARAMS = CN()
#     cfg.ConvNeXt.PARAMS.num_classes = 1000
#     cfg.ConvNeXt.PARAMS.in_channels = 3
#     cfg.ConvNeXt.PARAMS.stem_out_channels = 96
#     cfg.ConvNeXt.PARAMS.stem_kernel_size = 4
#     cfg.ConvNeXt.PARAMS.stem_kernel_stride = 4
#     cfg.ConvNeXt.PARAMS.num_blocks = [3, 3, 9, 3]
#     cfg.ConvNeXt.PARAMS.expansion_rate = 4
#     cfg.ConvNeXt.PARAMS.depthwise_conv_kernel_size = 7
#     cfg.ConvNeXt.PARAMS.layer_scale = 1e-6
#     cfg.ConvNeXt.PARAMS.stochastic_depth_mp = 0.1

#     cfg.TRAIN = CN()
#     cfg.TRAIN.PARAMS = CN()
#     cfg.TRAIN.PARAMS.epochs = 1000
#     cfg.TRAIN.PARAMS.gradient_accumulation = True
#     cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 4096
#     cfg.TRAIN.PARAMS.gradient_clipping = None
#     cfg.TRAIN.PARAMS.exponential_moving_average = CN()
#     cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.999
#     cfg.TRAIN.PARAMS.exponential_moving_average.warmup_steps = 0
#     cfg.TRAIN.PARAMS.exponential_moving_average.decay_period = 1
#     cfg.TRAIN.PARAMS.exponential_moving_average.decay_method = "constant"

#     cfg.DATA_MIXING.enabled = True
#     cfg.DATA_MIXING.mixup_alpha = 0.2
#     cfg.DATA_MIXING.cutmix_alpha = 1.0
#     cfg.DATA_MIXING.cutmix_minmax = None
#     cfg.DATA_MIXING.prob = 0.5
#     cfg.DATA_MIXING.switch_prob = 0.5
#     cfg.DATA_MIXING.mode = "batch"
#     cfg.DATA_MIXING.correct_lam = True
#     cfg.DATA_MIXING.label_smoothing = 0.0
#     cfg.DATA_MIXING.one_hot_targets = False
#     cfg.DATA_MIXING.num_classes = 1000

#     cfg.PIPELINES = CN()
#     cfg.PIPELINES.TRAIN = [
#         dict(func="readImage", params=dict(uint8=True)),
#     ]
#     cfg.PIPELINES.VAL = [
#         dict(func="readImage", params=dict(uint8=True)),
#         dict(func="resizeWithAspectRatio", params=dict(size=256))
#     ]

#     cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS = CN()
#     cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.batch_size = 32
#     cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.shuffle = True
#     cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.num_workers = 8
#     cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.pin_memory = True
#     cfg.ConvNeXt.DATALOADER_TRAIN_PARAMS.drop_last = True

#     cfg.ConvNeXt.DATALOADER_VAL_PARAMS = CN()
#     cfg.ConvNeXt.DATALOADER_VAL_PARAMS.batch_size = 64
#     cfg.ConvNeXt.DATALOADER_VAL_PARAMS.shuffle = True
#     cfg.ConvNeXt.DATALOADER_VAL_PARAMS.num_workers = 8
#     cfg.ConvNeXt.DATALOADER_VAL_PARAMS.pin_memory = True
#     cfg.ConvNeXt.DATALOADER_VAL_PARAMS.drop_last = True

#     cfg.ConvNeXt.TRANSFORMS = CN()

#     cfg.ConvNeXt.TRANSFORMS.TRAIN = [
#         dict(name="ToPILImage", params=None),
#         dict(name="RandomResizedCrop", params=dict(size=(224, 224))),
#         dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
#         dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
#         dict(name="ToTensor", params=None),
#         dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
#         dict(name="RandomCutOut", params=dict(probability=0.1, mode="pixel", device="cpu")) # to be used after normalization when mode is pixel, as it will avoid changing the image statistics
#     ]

#     cfg.ConvNeXt.TRANSFORMS.VAL = [
#         dict(name="ToPILImage", params=None),
#         dict(name="CenterCrop", params=dict(size=(256, 256))),
#         dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
#         dict(name="ToTensor", params=None),
#         dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#     ]

#     cfg.ConvNeXt.OPTIMIZER = CN()
#     cfg.ConvNeXt.OPTIMIZER.NAME = "AdamW"
#     cfg.ConvNeXt.OPTIMIZER.PARAMS = CN()
#     cfg.ConvNeXt.OPTIMIZER.PARAMS.lr = 4e-3
#     cfg.ConvNeXt.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
#     cfg.ConvNeXt.OPTIMIZER.PARAMS.weight_decay = 0.05

#     cfg.ConvNeXt.LR_SCHEDULER = CN()
#     cfg.ConvNeXt.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS = CN()
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.lr_initial = 0.0
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.lr_final = cfg.ConvNeXt.OPTIMIZER.PARAMS.lr
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.warmup_epochs = 10
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.warmup_method = "linear"
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler = CN()
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 4
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 0.0
#     cfg.ConvNeXt.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
#     cfg.ConvNeXt.LOSS = CN()
#     cfg.ConvNeXt.LOSS.NAME = "CrossEntropyLoss"
#     cfg.ConvNeXt.LOSS.PARAMS = CN()
#     cfg.ConvNeXt.LOSS.PARAMS.weight = None
#     cfg.ConvNeXt.LOSS.PARAMS.size_average = None
#     cfg.ConvNeXt.LOSS.PARAMS.ignore_index = -100
#     cfg.ConvNeXt.LOSS.PARAMS.reduce = None
#     cfg.ConvNeXt.LOSS.PARAMS.label_smoothing = 0.1
#     cfg.ConvNeXt.LOSS.PARAMS.reduction = "mean"

#     cfg.REGULARIZATION = CN()
#     cfg.REGULARIZATION.MODE = ''
#     cfg.REGULARIZATION.STRENGTH = 0