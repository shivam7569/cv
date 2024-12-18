def HPool_ViTConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.HPool_ViT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "HPool_ViT"
    cfg.METRICS.NAME = "HPool_ViT"
    cfg.CHECKPOINT.BASENAME = "HPool_ViT"
    cfg.TENSORBOARD.BASENAME = "HPool_ViT"
    cfg.PROFILER.BASENAME = "HPool_ViT/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.HPool_ViT.PARAMS = CN()
    cfg.HPool_ViT.PARAMS.num_classes = 1000
    cfg.HPool_ViT.PARAMS.d_model = 768
    cfg.HPool_ViT.PARAMS.image_size = 192
    cfg.HPool_ViT.PARAMS.patch_size = 16
    cfg.HPool_ViT.PARAMS.classifier_mlp_d = 2048
    cfg.HPool_ViT.PARAMS.encoder_mlp_d = 768 * 4
    cfg.HPool_ViT.PARAMS.encoder_num_heads = 12
    cfg.HPool_ViT.PARAMS.num_encoder_blocks = 12
    cfg.HPool_ViT.PARAMS.dropout = 0.0
    cfg.HPool_ViT.PARAMS.encoder_dropout = 0.0
    cfg.HPool_ViT.PARAMS.encoder_attention_dropout = 0.0
    cfg.HPool_ViT.PARAMS.encoder_projection_dropout = 0.0
    cfg.HPool_ViT.PARAMS.patchify_technique = "linear"
    cfg.HPool_ViT.PARAMS.stochastic_depth = True
    cfg.HPool_ViT.PARAMS.stochastic_depth_mp = 0.1
    cfg.HPool_ViT.PARAMS.layer_scale = 1e-4
    cfg.HPool_ViT.PARAMS.ln_order = "post"
    cfg.HPool_ViT.PARAMS.hvt_pool = [1, 5, 9]
    cfg.HPool_ViT.PARAMS.in_channels = 3

    cfg.TRAIN = CN()
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 500
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 1024
    cfg.TRAIN.PARAMS.gradient_clipping = None
    cfg.TRAIN.PARAMS.exponential_moving_average = CN()
    cfg.TRAIN.PARAMS.exponential_moving_average.beta = 0.999
    cfg.TRAIN.PARAMS.exponential_moving_average.update_every = 32
    cfg.TRAIN.PARAMS.exponential_moving_average.update_after_step = 0
    cfg.TRAIN.PARAMS.exponential_moving_average.inv_gamma = 1.0
    cfg.TRAIN.PARAMS.exponential_moving_average.power = 2/3
    cfg.TRAIN.PARAMS.exponential_moving_average.use_foreach = True
    cfg.TRAIN.PARAMS.exponential_moving_average.include_online_model = False
    cfg.TRAIN.PARAMS.exponential_moving_average.update_model_with_ema_every = 32*1000
    cfg.TRAIN.PARAMS.exponential_moving_average.update_model_with_ema_beta = 0.0
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
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.HPool_ViT.PARAMS.image_size == 192 else 256))
    ]

    cfg.HPool_ViT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.HPool_ViT.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.HPool_ViT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.HPool_ViT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.HPool_ViT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.HPool_ViT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.HPool_ViT.DATALOADER_VAL_PARAMS = CN()
    cfg.HPool_ViT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.HPool_ViT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.HPool_ViT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.HPool_ViT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.HPool_ViT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.HPool_ViT.TRANSFORMS = CN()

    cfg.HPool_ViT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.HPool_ViT.PARAMS.image_size, cfg.HPool_ViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.HPool_ViT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.HPool_ViT.PARAMS.image_size, cfg.HPool_ViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.HPool_ViT.OPTIMIZER = CN()
    cfg.HPool_ViT.OPTIMIZER.NAME = "AdamW"
    cfg.HPool_ViT.OPTIMIZER.PARAMS = CN()
    cfg.HPool_ViT.OPTIMIZER.PARAMS.lr = 5e-4 * cfg.num_gpus 
    cfg.HPool_ViT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.HPool_ViT.OPTIMIZER.PARAMS.weight_decay = 0.025

    cfg.HPool_ViT.LR_SCHEDULER = CN()
    cfg.HPool_ViT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS = CN()
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.lr_final = cfg.HPool_ViT.OPTIMIZER.PARAMS.lr
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.HPool_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.HPool_ViT.LOSS = CN()
    cfg.HPool_ViT.LOSS.NAME = "CrossEntropyLoss"
    cfg.HPool_ViT.LOSS.PARAMS = CN()
    cfg.HPool_ViT.LOSS.PARAMS.weight = None
    cfg.HPool_ViT.LOSS.PARAMS.size_average = None
    cfg.HPool_ViT.LOSS.PARAMS.ignore_index = -100
    cfg.HPool_ViT.LOSS.PARAMS.reduce = None
    cfg.HPool_ViT.LOSS.PARAMS.reduction = "mean"
    cfg.HPool_ViT.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0