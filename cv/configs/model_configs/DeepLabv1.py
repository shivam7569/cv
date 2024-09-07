def DeepLabv1Config(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.DeepLabv1 = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = True
    cfg.WRITE_TENSORBOARD_GRAPH = True

    cfg.PROFILING = False
    cfg.EVALUATION_STEPS = 25

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "DeepLabv1"
    cfg.METRICS.NAME = "DeepLabv1"
    cfg.CHECKPOINT.BASENAME = "DeepLabv1"
    cfg.TENSORBOARD.BASENAME = "DeepLabv1"
    cfg.PROFILER.BASENAME = "DeepLabv1/Profiling"

    cfg.DeepLabv1.PARAMS = CN()
    cfg.DeepLabv1.PARAMS.num_classes = 81
    cfg.DeepLabv1.PARAMS.load_weights = False
    cfg.DeepLabv1.PARAMS.dropout_rate = 0.5
    cfg.DeepLabv1.PARAMS.backbone_params = CN()
    cfg.DeepLabv1.PARAMS.backbone_params.num_classes = 1000
    cfg.DeepLabv1.PARAMS.backbone_params.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 500
    cfg.TRAIN.PARAMS.gradient_accumulation = False
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = None
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.TRAIN.PARAMS.evaluation_util_class_or_function = CN()
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.NAME = "DeepLabEval"
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS = CN()
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.interpolation_params = CN()
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.interpolation_params.mode = "bilinear"
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.interpolation_params.align_corners = False
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.crf_params = CN()
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.crf_params.iter_max = 10
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.crf_params.pos_xy_std = 3
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.crf_params.pos_w = 3
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.crf_params.bi_xy_std = 140
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.crf_params.bi_rgb_std = 5
    cfg.TRAIN.PARAMS.evaluation_util_class_or_function.PARAMS.crf_params.bi_w = 5

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


    cfg.DeepLabv1.DATALOADER_TRAIN_PARAMS = CN()
    cfg.DeepLabv1.DATALOADER_TRAIN_PARAMS.batch_size = 32
    cfg.DeepLabv1.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.DeepLabv1.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.DeepLabv1.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.DeepLabv1.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.DeepLabv1.DATALOADER_VAL_PARAMS = CN()
    cfg.DeepLabv1.DATALOADER_VAL_PARAMS.batch_size = 16
    cfg.DeepLabv1.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.DeepLabv1.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.DeepLabv1.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.DeepLabv1.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.DeepLabv1.TRANSFORMS = CN()

    cfg.DeepLabv1.TRANSFORMS.TRAIN = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationRandomCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=cfg.COCO_MEAN, std=cfg.COCO_STD))
    ]

    cfg.DeepLabv1.TRANSFORMS.VAL = [
        dict(name="SegmentationToPILImage", params=dict()),
        dict(name="SegmentationCenterCrop", params=dict(size=(224, 224))),
        dict(name="SegmentationHorizontalFlip", params=dict(p=0.5)),
        dict(name="SegmentationToTensor", params=dict()),
        dict(name="SegmentationNormalize", params=dict(mean=cfg.COCO_MEAN, std=cfg.COCO_STD))
    ]

    cfg.DeepLabv1.OPTIMIZER = CN()
    cfg.DeepLabv1.OPTIMIZER.NAME = "SGD"
    cfg.DeepLabv1.OPTIMIZER.PARAMS = CN()
    cfg.DeepLabv1.OPTIMIZER.PARAMS.lr = CN()
    cfg.DeepLabv1.OPTIMIZER.PARAMS.lr.GLOBAL = 0.001
    cfg.DeepLabv1.OPTIMIZER.PARAMS.lr.PARTITION = CN() 
    cfg.DeepLabv1.OPTIMIZER.PARAMS.lr.PARTITION.NAMES = ["feature_extractor", "classifier"] 
    cfg.DeepLabv1.OPTIMIZER.PARAMS.lr.PARTITION.LRS = [0.001, 00.1]
    cfg.DeepLabv1.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.DeepLabv1.OPTIMIZER.PARAMS.weight_decay = 1e-8

    cfg.DeepLabv1.LR_SCHEDULER = CN()
    cfg.DeepLabv1.LR_SCHEDULER.NAME = "CosineAnnealingLR"
    cfg.DeepLabv1.LR_SCHEDULER.PARAMS = CN()
    cfg.DeepLabv1.LR_SCHEDULER.PARAMS.T_max = 500
    cfg.DeepLabv1.LR_SCHEDULER.PARAMS.eta_min = 1e-4
    cfg.DeepLabv1.LR_SCHEDULER.PARAMS.last_epoch = -1

    cfg.DeepLabv1.LOSS = CN()
    cfg.DeepLabv1.LOSS.NAME = "DeepLabv1Loss"
    cfg.DeepLabv1.LOSS.PARAMS = CN()
    cfg.DeepLabv1.LOSS.PARAMS.name = "combo"
    cfg.DeepLabv1.LOSS.PARAMS._lambda = 0.3
    cfg.DeepLabv1.LOSS.PARAMS.dynamic_weighting = False
    cfg.DeepLabv1.LOSS.PARAMS.focal_params = CN()
    cfg.DeepLabv1.LOSS.PARAMS.focal_params.gamma = 2
    cfg.DeepLabv1.LOSS.PARAMS.focal_params.focal_reduction = "mean"
    cfg.DeepLabv1.LOSS.PARAMS.focal_params.ignore_index = 255
    cfg.DeepLabv1.LOSS.PARAMS.focal_params.reduction = "none"
    cfg.DeepLabv1.LOSS.PARAMS.focal_params.label_smoothing = 0.0
    cfg.DeepLabv1.LOSS.PARAMS.focal_params.normalize = False
    cfg.DeepLabv1.LOSS.PARAMS.dice_params = CN()
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.num_classes = 81
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.ignore_index = 255
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.reduction = "mean"
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.log_loss = False
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.log_cosh = True
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.normalize = False
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.smooth = 0.0
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.classes = None
    cfg.DeepLabv1.LOSS.PARAMS.dice_params.normalize = False
    cfg.DeepLabv1.LOSS.PARAMS.class_weightage_method = "median_frequency"

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
