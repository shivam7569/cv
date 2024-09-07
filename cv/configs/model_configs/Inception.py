def InceptionConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ASYNC_TRAINING = True
    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 150
    cfg.TRAIN.PARAMS.phase_dependent = True

    cfg.Inception = CN()

    cfg.Inception.PARAMS = CN()
    cfg.Inception.PARAMS.num_classes = 1000
    cfg.Inception.PARAMS.in_channels = 3

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "Inception"
    cfg.METRICS.NAME = "Inception"
    cfg.CHECKPOINT.BASENAME = "Inception"
    cfg.TENSORBOARD.BASENAME = "Inception"

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="inceptionv1resize", params=dict(aspect_ratio_min=3/4, aspect_ratio_max=4/3))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256))
    ]

    cfg.Inception.DATALOADER_TRAIN_PARAMS = CN()
    cfg.Inception.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.Inception.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.Inception.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.Inception.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.Inception.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.Inception.DATALOADER_VAL_PARAMS = CN()
    cfg.Inception.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.Inception.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.Inception.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.Inception.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.Inception.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.Inception.TRANSFORMS = CN()
    cfg.Inception.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomApply", params=dict(
            transforms=[
                dict(name="ColorJitter", params=dict(brightness=[0, 1], contrast=[0, 1], saturation=[0, 1], hue=0.3))
            ], p=0.5
        )),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.Inception.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(224, 224))),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.Inception.OPTIMIZER = CN()
    cfg.Inception.OPTIMIZER.NAME = "SGD"
    cfg.Inception.OPTIMIZER.PARAMS = CN()
    cfg.Inception.OPTIMIZER.PARAMS.lr = 0.01
    cfg.Inception.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.Inception.OPTIMIZER.PARAMS.weight_decay = 0.0005

    cfg.Inception.LR_SCHEDULER = CN()
    cfg.Inception.LR_SCHEDULER.NAME = "StepLR"
    cfg.Inception.LR_SCHEDULER.PARAMS = CN()
    cfg.Inception.LR_SCHEDULER.PARAMS.step_size = 30
    cfg.Inception.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.Inception.LR_SCHEDULER.PARAMS.verbose = False

    cfg.Inception.LOSS = CN()
    cfg.Inception.LOSS.NAME = "InceptionLoss"
    cfg.Inception.LOSS.PARAMS = CN()
    cfg.Inception.LOSS.PARAMS.ce_loss_params = CN()
    cfg.Inception.LOSS.PARAMS.ce_loss_params.weight = None
    cfg.Inception.LOSS.PARAMS.ce_loss_params.size_average = None
    cfg.Inception.LOSS.PARAMS.ce_loss_params.ignore_index = -100
    cfg.Inception.LOSS.PARAMS.ce_loss_params.reduce = None
    cfg.Inception.LOSS.PARAMS.ce_loss_params.reduction = "mean"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0