def SegNetConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.SegNet = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = True
    cfg.WRITE_TENSORBOARD_GRAPH = True

    cfg.PROFILING = False
    cfg.EVALUATION_STEPS = 15

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "SegNet"
    cfg.METRICS.NAME = "SegNet"
    cfg.CHECKPOINT.BASENAME = "SegNet"
    cfg.TENSORBOARD.BASENAME = "SegNet"
    cfg.PROFILER.BASENAME = "SegNet/Profiling"

    cfg.SegNet.PARAMS = CN()
    cfg.SegNet.PARAMS.num_classes = 81
    cfg.SegNet.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 600
    cfg.TRAIN.PARAMS.gradient_accumulation = False
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = None
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="remove_bg", params=dict(threshold=5, size=256)),
        dict(func="scale_factor_resize", params=dict(scales=[0.5, 0.75, 1.0, 1.25, 1.5])),
        dict(func="mask_to_img_size", params=dict()),
        dict(func="fit_to_size", params=dict(size=256, padding=True, ignore_label=255))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256)),
        dict(func="mask_to_img_size", params=dict())
    ]


    cfg.SegNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.SegNet.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.SegNet.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.SegNet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.SegNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.SegNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.SegNet.DATALOADER_VAL_PARAMS = CN()
    cfg.SegNet.DATALOADER_VAL_PARAMS.batch_size = 16
    cfg.SegNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.SegNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.SegNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.SegNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.SegNet.TRANSFORMS = CN()

    cfg.SegNet.TRANSFORMS.TRAIN = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationRandomCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=cfg.COCO_MEAN, std=cfg.COCO_STD))
    ]

    cfg.SegNet.TRANSFORMS.VAL = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationCenterCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=cfg.COCO_MEAN, std=cfg.COCO_STD))
    ]

    cfg.SegNet.OPTIMIZER = CN()
    cfg.SegNet.OPTIMIZER.NAME = "SGD"
    cfg.SegNet.OPTIMIZER.PARAMS = CN()
    cfg.SegNet.OPTIMIZER.PARAMS.lr = 0.1
    cfg.SegNet.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.SegNet.OPTIMIZER.PARAMS.weight_decay = 1e-8

    cfg.SegNet.LR_SCHEDULER = CN()
    cfg.SegNet.LR_SCHEDULER.NAME = "CosineAnnealingLR"
    cfg.SegNet.LR_SCHEDULER.PARAMS = CN()
    cfg.SegNet.LR_SCHEDULER.PARAMS.T_max = 600
    cfg.SegNet.LR_SCHEDULER.PARAMS.eta_min = 1e-5
    cfg.SegNet.LR_SCHEDULER.PARAMS.last_epoch = -1

    cfg.SegNet.LOSS = CN()
    cfg.SegNet.LOSS.NAME = "ComboLoss"
    cfg.SegNet.LOSS.PARAMS = CN()
    cfg.SegNet.LOSS.PARAMS._lambda = 0.3
    cfg.SegNet.LOSS.PARAMS.dynamic_weighting = False
    cfg.SegNet.LOSS.PARAMS.focal_params = CN()
    cfg.SegNet.LOSS.PARAMS.focal_params.gamma = 2
    cfg.SegNet.LOSS.PARAMS.focal_params.focal_reduction = "mean"
    cfg.SegNet.LOSS.PARAMS.focal_params.ignore_index = 255
    cfg.SegNet.LOSS.PARAMS.focal_params.reduction = "none"
    cfg.SegNet.LOSS.PARAMS.focal_params.label_smoothing = 0.0
    cfg.SegNet.LOSS.PARAMS.focal_params.normalize = False
    cfg.SegNet.LOSS.PARAMS.dice_params = CN()
    cfg.SegNet.LOSS.PARAMS.dice_params.num_classes = 81
    cfg.SegNet.LOSS.PARAMS.dice_params.ignore_index = 255
    cfg.SegNet.LOSS.PARAMS.dice_params.reduction = "mean"
    cfg.SegNet.LOSS.PARAMS.dice_params.log_loss = False
    cfg.SegNet.LOSS.PARAMS.dice_params.log_cosh = True
    cfg.SegNet.LOSS.PARAMS.dice_params.normalize = False
    cfg.SegNet.LOSS.PARAMS.dice_params.smooth = 0.0
    cfg.SegNet.LOSS.PARAMS.dice_params.classes = None
    cfg.SegNet.LOSS.PARAMS.dice_params.normalize = False
    cfg.SegNet.LOSS.PARAMS.class_weightage_method = "median_frequency"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
