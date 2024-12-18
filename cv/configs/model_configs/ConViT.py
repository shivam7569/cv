def ConViTConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.ConViT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "ConViT"
    cfg.METRICS.NAME = "ConViT"
    cfg.CHECKPOINT.BASENAME = "ConViT"
    cfg.TENSORBOARD.BASENAME = "ConViT"
    cfg.PROFILER.BASENAME = "ConViT/Exp"
    cfg.CHECKPOINT.TREE = False

    cfg.ConViT.PARAMS = CN()
    cfg.ConViT.PARAMS.d_model = 1024
    cfg.ConViT.PARAMS.image_size = 224
    cfg.ConViT.PARAMS.patch_size = 16
    cfg.ConViT.PARAMS.classifier_mlp_d = 2048
    cfg.ConViT.PARAMS.encoder_mlp_d = 1024 * 4
    cfg.ConViT.PARAMS.encoder_num_heads = 16
    cfg.ConViT.PARAMS.num_encoder_blocks = 12
    cfg.ConViT.PARAMS.num_gated_blocks = 10
    cfg.ConViT.PARAMS.locality_strength = 1.0
    cfg.ConViT.PARAMS.locality_distance_method = "constant"
    cfg.ConViT.PARAMS.use_conv_init = True
    cfg.ConViT.PARAMS.d_pos = 3
    cfg.ConViT.PARAMS.dropout = 0.1
    cfg.ConViT.PARAMS.encoder_dropout = 0.1
    cfg.ConViT.PARAMS.encoder_attention_dropout = 0.1
    cfg.ConViT.PARAMS.encoder_projection_dropout = 0.1
    cfg.ConViT.PARAMS.patchify_technique = "linear"
    cfg.ConViT.PARAMS.stochastic_depth = True
    cfg.ConViT.PARAMS.stochastic_depth_mp = 0.1
    cfg.ConViT.PARAMS.layer_scale = 1e-4
    cfg.ConViT.PARAMS.ln_order = "post"
    cfg.ConViT.PARAMS.in_channels = 3
    cfg.ConViT.PARAMS.num_classes = 1000

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
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.ConViT.PARAMS.image_size == 192 else 256))
    ]

    cfg.ConViT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.ConViT.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.ConViT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.ConViT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.ConViT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.ConViT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.ConViT.DATALOADER_VAL_PARAMS = CN()
    cfg.ConViT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.ConViT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.ConViT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.ConViT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.ConViT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.ConViT.TRANSFORMS = CN()

    cfg.ConViT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.ConViT.PARAMS.image_size, cfg.ConViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.ConViT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.ConViT.PARAMS.image_size, cfg.ConViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.ConViT.OPTIMIZER = CN()
    cfg.ConViT.OPTIMIZER.NAME = "AdamW"
    cfg.ConViT.OPTIMIZER.PARAMS = CN()
    cfg.ConViT.OPTIMIZER.PARAMS.lr = 5e-4 * cfg.num_gpus 
    cfg.ConViT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.ConViT.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.ConViT.LR_SCHEDULER = CN()
    cfg.ConViT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.ConViT.LR_SCHEDULER.PARAMS = CN()
    cfg.ConViT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.ConViT.LR_SCHEDULER.PARAMS.lr_final = cfg.ConViT.OPTIMIZER.PARAMS.lr
    cfg.ConViT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.ConViT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.ConViT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.ConViT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.ConViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.ConViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.ConViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.ConViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.ConViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.ConViT.LOSS = CN()
    cfg.ConViT.LOSS.NAME = "CrossEntropyLoss"
    cfg.ConViT.LOSS.PARAMS = CN()
    cfg.ConViT.LOSS.PARAMS.weight = None
    cfg.ConViT.LOSS.PARAMS.size_average = None
    cfg.ConViT.LOSS.PARAMS.ignore_index = -100
    cfg.ConViT.LOSS.PARAMS.reduce = None
    cfg.ConViT.LOSS.PARAMS.reduction = "mean"
    cfg.ConViT.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0