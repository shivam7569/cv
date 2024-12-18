def CeiTConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.CeiT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "CeiT"
    cfg.METRICS.NAME = "CeiT"
    cfg.CHECKPOINT.BASENAME = "CeiT"
    cfg.TENSORBOARD.BASENAME = "CeiT"
    cfg.PROFILER.BASENAME = "CeiT/Profiling"

    cfg.CeiT.PARAMS = CN()
    cfg.CeiT.PARAMS.image_size = 224
    cfg.CeiT.PARAMS.d_model = 768
    cfg.CeiT.PARAMS.patch_size = 4
    cfg.CeiT.PARAMS.dropout = 0.0
    cfg.CeiT.PARAMS.encoder_num_heads = 12
    cfg.CeiT.PARAMS.num_encoder_blocks = 12
    cfg.CeiT.PARAMS.encoder_dropout = 0.0
    cfg.CeiT.PARAMS.encoder_attention_dropout = 0.0
    cfg.CeiT.PARAMS.encoder_projection_dropout = 0.0
    cfg.CeiT.PARAMS.classifier_mlp_d = 2048
    cfg.CeiT.PARAMS.i2t_out_channels = 32
    cfg.CeiT.PARAMS.i2t_conv_kernel_size = 7
    cfg.CeiT.PARAMS.i2t_conv_stride = 2
    cfg.CeiT.PARAMS.i2t_max_pool_kernel_size = 3
    cfg.CeiT.PARAMS.i2t_max_pool_stride = 2
    cfg.CeiT.PARAMS.leff_expand_ratio = 4
    cfg.CeiT.PARAMS.leff_depthwise_kernel = 3
    cfg.CeiT.PARAMS.leff_depthwise_stride = 1
    cfg.CeiT.PARAMS.leff_depthwise_padding = 1
    cfg.CeiT.PARAMS.leff_depthwise_separable = True
    cfg.CeiT.PARAMS.lca_encoder_expansion_ratio = 4
    cfg.CeiT.PARAMS.lca_encoder_num_heads = 12
    cfg.CeiT.PARAMS.lca_encoder_dropout = 0.0
    cfg.CeiT.PARAMS.lca_encoder_attention_dropout = 0.0
    cfg.CeiT.PARAMS.lca_encoder_projection_dropout = 0.0
    cfg.CeiT.PARAMS.lca_encoder_ln_order = "post"
    cfg.CeiT.PARAMS.lca_encoder_stodepth_prob = 0.0
    cfg.CeiT.PARAMS.lca_encoder_layer_scale = None
    cfg.CeiT.PARAMS.lca_encoder_qkv_bias = False
    cfg.CeiT.PARAMS.patchify_technique = "linear"
    cfg.CeiT.PARAMS.stochastic_depth = True
    cfg.CeiT.PARAMS.stochastic_depth_mp = 0.1
    cfg.CeiT.PARAMS.layer_scale = 1e-4
    cfg.CeiT.PARAMS.ln_order = "post"
    cfg.CeiT.PARAMS.num_classes = 1000
    cfg.CeiT.PARAMS.in_channels = 3

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
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.CeiT.PARAMS.image_size == 192 else 256))
    ]

    cfg.CeiT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.CeiT.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.CeiT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.CeiT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.CeiT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.CeiT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.CeiT.DATALOADER_VAL_PARAMS = CN()
    cfg.CeiT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.CeiT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.CeiT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.CeiT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.CeiT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.CeiT.TRANSFORMS = CN()

    cfg.CeiT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.CeiT.PARAMS.image_size, cfg.CeiT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.CeiT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.CeiT.PARAMS.image_size, cfg.CeiT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.CeiT.OPTIMIZER = CN()
    cfg.CeiT.OPTIMIZER.NAME = "AdamW"
    cfg.CeiT.OPTIMIZER.PARAMS = CN()
    cfg.CeiT.OPTIMIZER.PARAMS.lr = 1e-3 * cfg.num_gpus 
    cfg.CeiT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.CeiT.OPTIMIZER.PARAMS.weight_decay = 0.05

    cfg.CeiT.LR_SCHEDULER = CN()
    cfg.CeiT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.CeiT.LR_SCHEDULER.PARAMS = CN()
    cfg.CeiT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.CeiT.LR_SCHEDULER.PARAMS.lr_final = cfg.CeiT.OPTIMIZER.PARAMS.lr
    cfg.CeiT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.CeiT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.CeiT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.CeiT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.CeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.CeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.CeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.CeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.CeiT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.CeiT.LOSS = CN()
    cfg.CeiT.LOSS.NAME = "CrossEntropyLoss"
    cfg.CeiT.LOSS.PARAMS = CN()
    cfg.CeiT.LOSS.PARAMS.weight = None
    cfg.CeiT.LOSS.PARAMS.size_average = None
    cfg.CeiT.LOSS.PARAMS.ignore_index = -100
    cfg.CeiT.LOSS.PARAMS.reduce = None
    cfg.CeiT.LOSS.PARAMS.reduction = "mean"
    cfg.CeiT.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
