def SENetConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.SENet = CN()

    cfg.ASYNC_TRAINING = True
    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "SENet"
    cfg.METRICS.NAME = "SENet"
    cfg.CHECKPOINT.BASENAME = "SENet"
    cfg.TENSORBOARD.BASENAME = "SENet"

    cfg.SENet.PARAMS = CN()
    cfg.SENet.PARAMS.num_classes = 1000
    cfg.SENet.PARAMS.in_channels = 3
    cfg.SENet.PARAMS.reduction_ratio = 16

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 150
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 1024

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.SENet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.SENet.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.SENet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.SENet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.SENet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.SENet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.SENet.DATALOADER_VAL_PARAMS = CN()
    cfg.SENet.DATALOADER_VAL_PARAMS.batch_size = 32
    cfg.SENet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.SENet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.SENet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.SENet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.SENet.TRANSFORMS = CN()
    cfg.SENet.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.SENet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.SENet.OPTIMIZER = CN()
    cfg.SENet.OPTIMIZER.NAME = "SGD"
    cfg.SENet.OPTIMIZER.PARAMS = CN()
    cfg.SENet.OPTIMIZER.PARAMS.lr = 0.6 * cfg.num_gpus
    cfg.SENet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.SENet.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.SENet.LR_SCHEDULER = CN()
    cfg.SENet.LR_SCHEDULER.NAME = "StepLR"
    cfg.SENet.LR_SCHEDULER.PARAMS = CN()
    cfg.SENet.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.SENet.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.SENet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.SENet.LOSS = CN()
    cfg.SENet.LOSS.NAME = "CrossEntropyLoss"
    cfg.SENet.LOSS.PARAMS = CN()
    cfg.SENet.LOSS.PARAMS.weight = None
    cfg.SENet.LOSS.PARAMS.size_average = None
    cfg.SENet.LOSS.PARAMS.ignore_index = -100
    cfg.SENet.LOSS.PARAMS.reduce = None
    cfg.SENet.LOSS.PARAMS.reduction = "mean"
    cfg.SENet.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0