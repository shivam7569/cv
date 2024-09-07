def ResNeXtConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ResNeXt = CN()

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ResNeXt"
    cfg.METRICS.NAME = "ResNeXt"
    cfg.CHECKPOINT.BASENAME = "ResNeXt"
    cfg.TENSORBOARD.BASENAME = "ResNeXt"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv1resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.batch_size = 64
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ResNeXt.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ResNeXt.DATALOADER_VAL_PARAMS = CN()
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ResNeXt.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ResNeXt.TRANSFORMS = CN()
    cfg.ResNeXt.TRANSFORMS.TRAIN = [
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
    cfg.ResNeXt.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ResNeXt.OPTIMIZER = CN()
    cfg.ResNeXt.OPTIMIZER.NAME = "SGD"
    cfg.ResNeXt.OPTIMIZER.PARAMS = CN()
    cfg.ResNeXt.OPTIMIZER.PARAMS.lr = 0.1
    cfg.ResNeXt.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.ResNeXt.OPTIMIZER.PARAMS.weight_decay = 1e-4

    cfg.ResNeXt.LR_SCHEDULER = CN()
    cfg.ResNeXt.LR_SCHEDULER.NAME = "StepLR"
    cfg.ResNeXt.LR_SCHEDULER.PARAMS = CN()
    cfg.ResNeXt.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.ResNeXt.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.ResNeXt.LR_SCHEDULER.PARAMS.verbose = False

    cfg.ResNeXt.LOSS = CN()
    cfg.ResNeXt.LOSS.NAME = "CrossEntropyLoss"
    cfg.ResNeXt.LOSS.PARAMS = CN()
    cfg.ResNeXt.LOSS.PARAMS.weight = None
    cfg.ResNeXt.LOSS.PARAMS.size_average = None
    cfg.ResNeXt.LOSS.PARAMS.ignore_index = -100
    cfg.ResNeXt.LOSS.PARAMS.reduce = None
    cfg.ResNeXt.LOSS.PARAMS.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0