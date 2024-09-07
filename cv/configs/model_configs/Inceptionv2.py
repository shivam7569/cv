def Inceptionv2Config(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.Inceptionv2 = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "Inceptionv2"
    cfg.METRICS.NAME = "Inceptionv2"
    cfg.CHECKPOINT.BASENAME = "Inceptionv2"
    cfg.TENSORBOARD.BASENAME = "Inceptionv2"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv2resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=340))
    ]

    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS = CN()
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.Inceptionv2.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.Inceptionv2.DATALOADER_VAL_PARAMS = CN()
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.Inceptionv2.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.Inceptionv2.TRANSFORMS = CN()
    cfg.Inceptionv2.TRANSFORMS.TRAIN = [
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
    cfg.Inceptionv2.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(299, 299))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.Inceptionv2.OPTIMIZER = CN()
    cfg.Inceptionv2.OPTIMIZER.NAME = "SGD"
    cfg.Inceptionv2.OPTIMIZER.PARAMS = CN()
    cfg.Inceptionv2.OPTIMIZER.PARAMS.lr = 0.01
    cfg.Inceptionv2.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.Inceptionv2.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.Inceptionv2.LR_SCHEDULER = CN()
    cfg.Inceptionv2.LR_SCHEDULER.NAME = "StepLR"
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS = CN()
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.Inceptionv2.LR_SCHEDULER.PARAMS.verbose = False

    cfg.Inceptionv2.LOSS = CN()
    cfg.Inceptionv2.LOSS.NAME = "CrossEntropyLoss"
    cfg.Inceptionv2.LOSS.PARAMS = CN()
    cfg.Inceptionv2.LOSS.PARAMS.weight = None
    cfg.Inceptionv2.LOSS.PARAMS.size_average = None
    cfg.Inceptionv2.LOSS.PARAMS.ignore_index = -100
    cfg.Inceptionv2.LOSS.PARAMS.reduce = None
    cfg.Inceptionv2.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0