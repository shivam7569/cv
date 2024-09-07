def Inceptionv4Config(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.Inceptionv4 = CN()

    cfg.ASYNC_TRAINING = True
    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "Inceptionv4"
    cfg.METRICS.NAME = "Inceptionv4"
    cfg.CHECKPOINT.BASENAME = "Inceptionv4"
    cfg.TENSORBOARD.BASENAME = "Inceptionv4"

    cfg.Inceptionv4.PARAMS = CN()
    cfg.Inceptionv4.PARAMS.blocks = [4, 7, 3]
    cfg.Inceptionv4.PARAMS.num_classes = 1000
    cfg.Inceptionv4.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 150
    cfg.TRAIN.PARAMS.lr_scheduler_step = 2
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 240
    cfg.TRAIN.PARAMS.gradient_clipping = None

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv2resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=324))
    ]

    cfg.Inceptionv4.DATALOADER_TRAIN_PARAMS = CN()
    cfg.Inceptionv4.DATALOADER_TRAIN_PARAMS.batch_size = 48
    cfg.Inceptionv4.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.Inceptionv4.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.Inceptionv4.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.Inceptionv4.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.Inceptionv4.DATALOADER_VAL_PARAMS = CN()
    cfg.Inceptionv4.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.Inceptionv4.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.Inceptionv4.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.Inceptionv4.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.Inceptionv4.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.Inceptionv4.TRANSFORMS = CN()
    cfg.Inceptionv4.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=[0, 1], contrast=[0, 1], saturation=[0, 1], hue=0.3))
            ], p=0.5
        )),
        dict(name="RandomCrop", params=dict(size=(299, 299), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.Inceptionv4.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(299, 299))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.Inceptionv4.OPTIMIZER = CN()
    cfg.Inceptionv4.OPTIMIZER.NAME = "RMSprop"
    cfg.Inceptionv4.OPTIMIZER.PARAMS = CN()
    cfg.Inceptionv4.OPTIMIZER.PARAMS.lr = 0.045
    cfg.Inceptionv4.OPTIMIZER.PARAMS.alpha = 0.9
    cfg.Inceptionv4.OPTIMIZER.PARAMS.eps = 1.0

    cfg.Inceptionv4.LR_SCHEDULER = CN()
    cfg.Inceptionv4.LR_SCHEDULER.NAME = "ExponentialLR"
    cfg.Inceptionv4.LR_SCHEDULER.PARAMS = CN()
    cfg.Inceptionv4.LR_SCHEDULER.PARAMS.gamma = 0.94
    cfg.Inceptionv4.LR_SCHEDULER.PARAMS.verbose = False

    cfg.Inceptionv4.LOSS = CN()
    cfg.Inceptionv4.LOSS.NAME = "CrossEntropyLoss"
    cfg.Inceptionv4.LOSS.PARAMS = CN()
    cfg.Inceptionv4.LOSS.PARAMS.weight = None
    cfg.Inceptionv4.LOSS.PARAMS.size_average = None
    cfg.Inceptionv4.LOSS.PARAMS.ignore_index = -100
    cfg.Inceptionv4.LOSS.PARAMS.reduce = None
    cfg.Inceptionv4.LOSS.PARAMS.reduction = "mean"
    cfg.Inceptionv4.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0