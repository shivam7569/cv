def MobileNetConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.MobileNet = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "MobileNet"
    cfg.METRICS.NAME = "MobileNet"
    cfg.CHECKPOINT.BASENAME = "MobileNet"
    cfg.TENSORBOARD.BASENAME = "MobileNet"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.MobileNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.MobileNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.MobileNet.DATALOADER_VAL_PARAMS = CN()
    cfg.MobileNet.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.MobileNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.MobileNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.MobileNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.MobileNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.MobileNet.TRANSFORMS = CN()
    cfg.MobileNet.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4))
            ], p=0.5
        )),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.MobileNet.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.MobileNet.OPTIMIZER = CN()
    cfg.MobileNet.OPTIMIZER.NAME = "RMSprop"
    cfg.MobileNet.OPTIMIZER.PARAMS = CN()
    cfg.MobileNet.OPTIMIZER.PARAMS.lr = 0.01
    cfg.MobileNet.OPTIMIZER.PARAMS.alpha = 0.99
    cfg.MobileNet.OPTIMIZER.PARAMS.eps = 1e-8
    cfg.MobileNet.OPTIMIZER.PARAMS.weight_decay = 0
    cfg.MobileNet.OPTIMIZER.PARAMS.momentum = 1e-5

    cfg.MobileNet.LR_SCHEDULER = CN()
    cfg.MobileNet.LR_SCHEDULER.NAME = "MultiplicativeLR"
    cfg.MobileNet.LR_SCHEDULER.FACTOR = 0.98
    cfg.MobileNet.LR_SCHEDULER.PARAMS = CN()
    cfg.MobileNet.LR_SCHEDULER.PARAMS.verbose = False

    cfg.MobileNet.LOSS = CN()
    cfg.MobileNet.LOSS.NAME = "CrossEntropyLoss"
    cfg.MobileNet.LOSS.PARAMS = CN()
    cfg.MobileNet.LOSS.PARAMS.weight = None
    cfg.MobileNet.LOSS.PARAMS.size_average = None
    cfg.MobileNet.LOSS.PARAMS.ignore_index = -100
    cfg.MobileNet.LOSS.PARAMS.reduce = None
    cfg.MobileNet.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0