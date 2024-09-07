def Inceptionv3Config(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.Inceptionv3 = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "Inceptionv3"
    cfg.METRICS.NAME = "Inceptionv3"
    cfg.CHECKPOINT.BASENAME = "Inceptionv3"
    cfg.TENSORBOARD.BASENAME = "Inceptionv3"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv2resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=340))
    ]

    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS = CN()
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.Inceptionv3.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.Inceptionv3.DATALOADER_VAL_PARAMS = CN()
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.Inceptionv3.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.Inceptionv3.TRANSFORMS = CN()
    cfg.Inceptionv3.TRANSFORMS.TRAIN = [
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
    cfg.Inceptionv3.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(299, 299))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.Inceptionv3.OPTIMIZER = CN()
    cfg.Inceptionv3.OPTIMIZER.NAME = "SGD"
    cfg.Inceptionv3.OPTIMIZER.PARAMS = CN()
    cfg.Inceptionv3.OPTIMIZER.PARAMS.lr = 0.1
    cfg.Inceptionv3.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.Inceptionv3.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.Inceptionv3.LR_SCHEDULER = CN()
    cfg.Inceptionv3.LR_SCHEDULER.NAME = "ExponentialLR"
    cfg.Inceptionv3.LR_SCHEDULER.PARAMS = CN()
    cfg.Inceptionv3.LR_SCHEDULER.PARAMS.gamma = 0.94
    cfg.Inceptionv3.LR_SCHEDULER.PARAMS.verbose = False

    cfg.Inceptionv3.LOSS = CN()
    cfg.Inceptionv3.LOSS.NAME = "CrossEntropyLoss"
    cfg.Inceptionv3.LOSS.PARAMS = CN()
    cfg.Inceptionv3.LOSS.PARAMS.weight = None
    cfg.Inceptionv3.LOSS.PARAMS.size_average = None
    cfg.Inceptionv3.LOSS.PARAMS.ignore_index = -100
    cfg.Inceptionv3.LOSS.PARAMS.reduce = None
    cfg.Inceptionv3.LOSS.PARAMS.reduction = "mean"
    cfg.Inceptionv3.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0