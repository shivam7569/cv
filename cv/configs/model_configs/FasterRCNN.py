def FasterRCNNConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.FasterRCNN = CN()

    cfg.ASYNC_TRAINING = False
    cfg.REPEAT_AUGMENTATIONS = False
    cfg.SAVE_FIRST_SAMPLE = False
    cfg.WRITE_TENSORBOARD_GRAPH = False

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "FasterRCNN"
    cfg.METRICS.NAME = "FasterRCNN"
    cfg.CHECKPOINT.BASENAME = "FasterRCNN"
    cfg.TENSORBOARD.BASENAME = "FasterRCNN"
    cfg.PROFILER.BASENAME = "FasterRCNN/Profiling"

    cfg.FasterRCNN.Stage1 = CN()
    cfg.FasterRCNN.Stage1.PARAMS = CN()
    cfg.FasterRCNN.Stage1.PARAMS.backbone_name = "VGG16"
    cfg.FasterRCNN.Stage1.PARAMS.backbone_params = CN()
    cfg.FasterRCNN.Stage1.PARAMS.backbone_params.num_classes = 1000
    cfg.FasterRCNN.Stage1.PARAMS.anchor_box_boundary_threshold = 1.0
    cfg.FasterRCNN.Stage1.PARAMS.anchor_box_positive_threshold = 0.7
    
    # cfg.FasterRCNN.PARAMS.patch_size = 4
    # cfg.FasterRCNN.PARAMS.window_size = 7
    # cfg.FasterRCNN.PARAMS.d_ff_ratio = 4
    # cfg.FasterRCNN.PARAMS.num_heads_per_stage = [4, 8, 16, 32]
    # cfg.FasterRCNN.PARAMS.shift = 2
    # cfg.FasterRCNN.PARAMS.patch_merge_size = 2
    # cfg.FasterRCNN.PARAMS.stage_embed_dim_ratio = 2
    # cfg.FasterRCNN.PARAMS.num_blocks = [2, 2, 18, 2]
    # cfg.FasterRCNN.PARAMS.classifier_mlp_d = 2048
    # cfg.FasterRCNN.PARAMS.encoder_dropout = 0.0
    # cfg.FasterRCNN.PARAMS.attention_dropout = 0.0
    # cfg.FasterRCNN.PARAMS.projection_dropout = 0.0
    # cfg.FasterRCNN.PARAMS.classifier_dropout = 0.0
    # cfg.FasterRCNN.PARAMS.global_aggregate = "avg"
    # cfg.FasterRCNN.PARAMS.image_size = 224
    # cfg.FasterRCNN.PARAMS.patchify_technique = "convolutional"
    # cfg.FasterRCNN.PARAMS.stodepth = True
    # cfg.FasterRCNN.PARAMS.stodepth_mp = 0.1
    # cfg.FasterRCNN.PARAMS.layer_scale = 1e-6
    # cfg.FasterRCNN.PARAMS.qkv_bias = False
    # cfg.FasterRCNN.PARAMS.in_channels = 3
    # cfg.FasterRCNN.PARAMS.num_classes = 1000

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 1000
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 1024
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = None
    cfg.TRAIN.PARAMS.updateStochasticDepthRate = None

    cfg.DATA_MIXING.enabled = True
    cfg.DATA_MIXING.mixup_alpha = 0.2
    cfg.DATA_MIXING.cutmix_alpha = 1.0
    cfg.DATA_MIXING.cutmix_minmax = None
    cfg.DATA_MIXING.prob = 0.8
    cfg.DATA_MIXING.switch_prob = 0.5
    cfg.DATA_MIXING.mode = "batch"
    cfg.DATA_MIXING.correct_lam = True
    cfg.DATA_MIXING.label_smoothing = 0.0
    cfg.DATA_MIXING.one_hot_targets = False
    cfg.DATA_MIXING.num_classes = 1000

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=600))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=600))
    ]

    cfg.FasterRCNN.DATALOADER_TRAIN_PARAMS = CN()
    cfg.FasterRCNN.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.FasterRCNN.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.FasterRCNN.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.FasterRCNN.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.FasterRCNN.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.FasterRCNN.DATALOADER_VAL_PARAMS = CN()
    cfg.FasterRCNN.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.FasterRCNN.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.FasterRCNN.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.FasterRCNN.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.FasterRCNN.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.FasterRCNN.TRANSFORMS = CN()

    cfg.FasterRCNN.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(600, 1000))),
        dict(name="DetectionHorizontalFlip", params=dict(p=0.5, normalized_anns=True)),
        # dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        # dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.FasterRCNN.TRANSFORMS.VAL = [
        dict(name="DetectionToPILImage", params=dict()),
        dict(name="DetectionResize", params=dict(size=(600, 1000), normalized_anns=True)),
        dict(name="DetectionHorizontalFlip", params=dict(p=0.5, normalized_anns=True)),
        dict(name="DetectionToTensor", params=dict()),
        dict(name="DetectionNormalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.FasterRCNN.OPTIMIZER = CN()
    cfg.FasterRCNN.OPTIMIZER.NAME = "AdamW"
    cfg.FasterRCNN.OPTIMIZER.PARAMS = CN()
    cfg.FasterRCNN.OPTIMIZER.PARAMS.lr = 1e-3 * cfg.num_gpus 
    cfg.FasterRCNN.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.FasterRCNN.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.FasterRCNN.LR_SCHEDULER = CN()
    cfg.FasterRCNN.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS = CN()
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.lr_final = cfg.FasterRCNN.OPTIMIZER.PARAMS.lr
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.warmup_epochs = 20
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.FasterRCNN.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.FasterRCNN.LOSS = CN()
    cfg.FasterRCNN.LOSS.NAME = "CrossEntropyLoss"
    cfg.FasterRCNN.LOSS.PARAMS = CN()
    cfg.FasterRCNN.LOSS.PARAMS.weight = None
    cfg.FasterRCNN.LOSS.PARAMS.size_average = None
    cfg.FasterRCNN.LOSS.PARAMS.ignore_index = -100
    cfg.FasterRCNN.LOSS.PARAMS.reduce = None
    cfg.FasterRCNN.LOSS.PARAMS.reduction = "mean"
    cfg.FasterRCNN.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
