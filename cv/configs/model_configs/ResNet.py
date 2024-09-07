def ResNetConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ResNet = CN()

    cfg.ASYNC_TRAINING = True

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ResNet"
    cfg.METRICS.NAME = "ResNet"
    cfg.CHECKPOINT.BASENAME = "ResNet"
    cfg.TENSORBOARD.BASENAME = "ResNet"

    cfg.REPEAT_AUGMENTATIONS = True
    cfg.REPEAT_AUGMENTATIONS_NUM_REPEATS = 4

    cfg.ResNet.PARAMS = CN()
    cfg.ResNet.PARAMS.num_classes = 1000
    cfg.ResNet.PARAMS.num_blocks = [3, 4, 23, 3]
    cfg.ResNet.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 600
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 512
    cfg.TRAIN.PARAMS.exponential_moving_average = CN()
    cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.99998
    cfg.TRAIN.PARAMS.exponential_moving_average.warmup_steps = 0
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_period = 32
    cfg.TRAIN.PARAMS.exponential_moving_average.decay_method = "constant"

    cfg.DATA_MIXING.enabled = True
    cfg.DATA_MIXING.mixup_alpha = 0.2
    cfg.DATA_MIXING.cutmix_alpha = 1.0
    cfg.DATA_MIXING.cutmix_minmax = None
    cfg.DATA_MIXING.prob = 0.8
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

    cfg.ResNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.batch_size = 64
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ResNet.DATALOADER_VAL_PARAMS = CN()
    cfg.ResNet.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.ResNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ResNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ResNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ResNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ResNet.TRANSFORMS = CN()
    cfg.ResNet.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(176, 176), pad_if_needed=False)),
        dict(name="TrivialAugmentWide", params=dict()),
        dict(name="RandomHorizontalFlip", params=dict(num_magnitude_bins=31)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.1, mode="pixel", device="cpu"))
    ]
    cfg.ResNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ResNet.OPTIMIZER = CN()
    cfg.ResNet.OPTIMIZER.NAME = "SGD"
    cfg.ResNet.OPTIMIZER.PARAMS = CN()
    cfg.ResNet.OPTIMIZER.PARAMS.lr = 0.5
    cfg.ResNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.ResNet.OPTIMIZER.PARAMS.weight_decay = 2e-5

    cfg.ResNet.LR_SCHEDULER = CN()
    cfg.ResNet.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.ResNet.LR_SCHEDULER.PARAMS = CN()
    cfg.ResNet.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.ResNet.LR_SCHEDULER.PARAMS.lr_final = cfg.ResNet.OPTIMIZER.PARAMS.lr
    cfg.ResNet.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.ResNet.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.ResNet.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.ResNet.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingLR"
    cfg.ResNet.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.ResNet.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_max = 595
    cfg.ResNet.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.ResNet.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1

    cfg.ResNet.LOSS = CN()
    cfg.ResNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.ResNet.LOSS.PARAMS = CN()
    cfg.ResNet.LOSS.PARAMS.weight = None
    cfg.ResNet.LOSS.PARAMS.size_average = None
    cfg.ResNet.LOSS.PARAMS.ignore_index = -100
    cfg.ResNet.LOSS.PARAMS.reduce = None
    cfg.ResNet.LOSS.PARAMS.reduction = "mean"
    cfg.ResNet.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0


"""
Original Paper Recipe:

def ResNetConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ResNet = CN()

    cfg.ASYNC_TRAINING = True

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ResNet"
    cfg.METRICS.NAME = "ResNet"
    cfg.CHECKPOINT.BASENAME = "ResNet"
    cfg.TENSORBOARD.BASENAME = "ResNet"

    cfg.ResNet.PARAMS = CN()
    cfg.ResNet.PARAMS.num_classes = 1000
    cfg.ResNet.PARAMS.num_blocks = [3, 4, 23, 3]
    cfg.ResNet.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 300
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 256

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="vgg16resize", params=dict(s_min=256, s_max=480))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.ResNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.batch_size = 64
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ResNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ResNet.DATALOADER_VAL_PARAMS = CN()
    cfg.ResNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.ResNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ResNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ResNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ResNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ResNet.TRANSFORMS = CN()
    cfg.ResNet.TRANSFORMS.TRAIN = [
        dict(name="FancyPCA", params=dict(alpha_std=0.1, p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4))
            ], p=0.5
        )),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.ResNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ResNet.OPTIMIZER = CN()
    cfg.ResNet.OPTIMIZER.NAME = "SGD"
    cfg.ResNet.OPTIMIZER.PARAMS = CN()
    cfg.ResNet.OPTIMIZER.PARAMS.lr = 0.1
    cfg.ResNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.ResNet.OPTIMIZER.PARAMS.weight_decay = 1e-4

    cfg.ResNet.LR_SCHEDULER = CN()
    cfg.ResNet.LR_SCHEDULER.NAME = "MultiplicativeLR"
    cfg.ResNet.LR_SCHEDULER.PARAMS = CN()
    cfg.ResNet.LR_SCHEDULER.PARAMS.factor = 0.945
    cfg.ResNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.ResNet.LOSS = CN()
    cfg.ResNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.ResNet.LOSS.PARAMS = CN()
    cfg.ResNet.LOSS.PARAMS.weight = None
    cfg.ResNet.LOSS.PARAMS.size_average = None
    cfg.ResNet.LOSS.PARAMS.ignore_index = -100
    cfg.ResNet.LOSS.PARAMS.reduce = None
    cfg.ResNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0

"""