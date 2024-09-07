def MobileNetv2Config(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.MobileNetv2 = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "MobileNetv2"
    cfg.METRICS.NAME = "MobileNetv2"
    cfg.CHECKPOINT.BASENAME = "MobileNetv2"
    cfg.TENSORBOARD.BASENAME = "MobileNetv2"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.MobileNetv2.DATALOADER_TRAIN_PARAMS = CN()
    cfg.MobileNetv2.DATALOADER_TRAIN_PARAMS.batch_size = 96
    cfg.MobileNetv2.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.MobileNetv2.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.MobileNetv2.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.MobileNetv2.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.MobileNetv2.DATALOADER_VAL_PARAMS = CN()
    cfg.MobileNetv2.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.MobileNetv2.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.MobileNetv2.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.MobileNetv2.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.MobileNetv2.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.MobileNetv2.TRANSFORMS = CN()
    cfg.MobileNetv2.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.MobileNetv2.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.MobileNetv2.OPTIMIZER = CN()
    cfg.MobileNetv2.OPTIMIZER.NAME = "RMSprop"
    cfg.MobileNetv2.OPTIMIZER.PARAMS = CN()
    cfg.MobileNetv2.OPTIMIZER.PARAMS.lr = 0.01
    cfg.MobileNetv2.OPTIMIZER.PARAMS.alpha = 0.99
    cfg.MobileNetv2.OPTIMIZER.PARAMS.eps = 1e-8
    cfg.MobileNetv2.OPTIMIZER.PARAMS.weight_decay = 0.00004
    cfg.MobileNetv2.OPTIMIZER.PARAMS.momentum = 0

    cfg.MobileNetv2.LR_SCHEDULER = CN()
    cfg.MobileNetv2.LR_SCHEDULER.NAME = "MultiplicativeLR"
    cfg.MobileNetv2.LR_SCHEDULER.FACTOR = 0.98
    cfg.MobileNetv2.LR_SCHEDULER.PARAMS = CN()
    cfg.MobileNetv2.LR_SCHEDULER.PARAMS.verbose = False

    cfg.MobileNetv2.LOSS = CN()
    cfg.MobileNetv2.LOSS.NAME = "CrossEntropyLoss"
    cfg.MobileNetv2.LOSS.PARAMS = CN()
    cfg.MobileNetv2.LOSS.PARAMS.weight = None
    cfg.MobileNetv2.LOSS.PARAMS.size_average = None
    cfg.MobileNetv2.LOSS.PARAMS.ignore_index = -100
    cfg.MobileNetv2.LOSS.PARAMS.reduce = None
    cfg.MobileNetv2.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0