def UNetConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.UNet = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = True
    cfg.WRITE_TENSORBOARD_GRAPH = True

    cfg.PROFILING = False
    cfg.SCHEDULE_FREE_TRAINING = True
    cfg.EVALUATION_STEPS = 20

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "UNet"
    cfg.METRICS.NAME = "UNet"
    cfg.CHECKPOINT.BASENAME = "UNet"
    cfg.TENSORBOARD.BASENAME = "UNet"
    cfg.PROFILER.BASENAME = "UNet/Profiling"

    cfg.UNet.PARAMS = CN()
    cfg.UNet.PARAMS.channels = [64, 128, 256, 512, 1024]
    cfg.UNet.PARAMS.in_channels = 3
    cfg.UNet.PARAMS.num_classes = 81
    cfg.UNet.PARAMS.dropout = 0.5
    cfg.UNet.PARAMS.retain_size = True

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 500
    cfg.TRAIN.PARAMS.gradient_accumulation = False
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = None
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="remove_bg", params=dict(threshold=2, size=256)),
        dict(func="resizeWithAspectRatio", params=dict(size=256)),
        dict(func="mask_to_img_size", params=dict())
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=256)),
        dict(func="mask_to_img_size", params=dict())
    ]

    cfg.UNet.DATALOADER_TRAIN_PARAMS = CN()
    cfg.UNet.DATALOADER_TRAIN_PARAMS.batch_size = 20
    cfg.UNet.DATALOADER_TRAIN_PARAMS.shuffle = False
    cfg.UNet.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.UNet.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.UNet.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.UNet.DATALOADER_VAL_PARAMS = CN()
    cfg.UNet.DATALOADER_VAL_PARAMS.batch_size = 16
    cfg.UNet.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.UNet.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.UNet.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.UNet.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.UNet.TRANSFORMS = CN()

    cfg.UNet.TRANSFORMS.TRAIN = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationElasticTransform", params=dict(p=0.05, alpha=50.0, sigma=5.0)),
        dict(name="SegmentationRandomCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationColorJitter", params=dict(brightness=0.4, contrast=0.4, saturation=0.4, p=0.05)),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=cfg.COCO_MEAN, std=cfg.COCO_STD))
    ]

    cfg.UNet.TRANSFORMS.VAL = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationCenterCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=cfg.COCO_MEAN, std=cfg.COCO_STD))
    ]

    cfg.UNet.OPTIMIZER = CN()
    cfg.UNet.OPTIMIZER.NAME = "SGD"
    cfg.UNet.OPTIMIZER.PARAMS = CN()
    cfg.UNet.OPTIMIZER.PARAMS.lr = 0.1
    cfg.UNet.OPTIMIZER.PARAMS.momentum = 0.99
    cfg.UNet.OPTIMIZER.PARAMS.weight_decay = 1e-8

    cfg.UNet.LR_SCHEDULER = CN()
    cfg.UNet.LR_SCHEDULER.NAME = "CosineAnnealingLR"
    cfg.UNet.LR_SCHEDULER.PARAMS = CN()
    cfg.UNet.LR_SCHEDULER.PARAMS.T_max = 500
    cfg.UNet.LR_SCHEDULER.PARAMS.eta_min = 1e-4
    cfg.UNet.LR_SCHEDULER.PARAMS.last_epoch = -1
    
    cfg.UNet.LOSS = CN()
    cfg.UNet.LOSS.NAME = "ComboLoss"
    cfg.UNet.LOSS.PARAMS = CN()
    cfg.UNet.LOSS.PARAMS._lambda = 0.5
    cfg.UNet.LOSS.PARAMS.focal_params = CN()
    cfg.UNet.LOSS.PARAMS.focal_params.gamma = 2
    cfg.UNet.LOSS.PARAMS.focal_params.focal_reduction = "mean"
    cfg.UNet.LOSS.PARAMS.focal_params.ignore_index = -1
    cfg.UNet.LOSS.PARAMS.focal_params.reduction = "none"
    cfg.UNet.LOSS.PARAMS.focal_params.label_smoothing = 0.0
    cfg.UNet.LOSS.PARAMS.dice_params = CN()
    cfg.UNet.LOSS.PARAMS.dice_params.num_classes = 81
    cfg.UNet.LOSS.PARAMS.dice_params.ignore_index = -1
    cfg.UNet.LOSS.PARAMS.dice_params.reduction = "mean"
    cfg.UNet.LOSS.PARAMS.dice_params.log_loss = False
    cfg.UNet.LOSS.PARAMS.dice_params.log_cosh = True
    cfg.UNet.LOSS.PARAMS.dice_params.normalize = False
    cfg.UNet.LOSS.PARAMS.dice_params.smooth = 0.0
    cfg.UNet.LOSS.PARAMS.dice_params.classes = None
    cfg.UNet.LOSS.PARAMS.class_weightage_method = "median_frequency"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
