def T2T_ViTConfig(cfg):

    from cv.configs.config import CfgNode as CN

    cfg.T2T_ViT = CN()

    cfg.ASYNC_TRAINING = True
    cfg.REPEAT_AUGMENTATIONS = True

    cfg.DEBUG = None
    cfg.PROFILING = False

    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    cfg.LOGGING.NAME = "T2T_ViT"
    cfg.METRICS.NAME = "T2T_ViT"
    cfg.CHECKPOINT.BASENAME = "T2T_ViT"
    cfg.TENSORBOARD.BASENAME = "T2T_ViT"
    cfg.PROFILER.BASENAME = "T2T_ViT/Profiling"

    cfg.T2T_ViT.PARAMS = CN()
    cfg.T2T_ViT.PARAMS.embed_dim = 384
    cfg.T2T_ViT.PARAMS.t2t_module_embed_dim = 64
    cfg.T2T_ViT.PARAMS.t2t_module_d_ff = 64
    cfg.T2T_ViT.PARAMS.t2t_module_transformer_num_heads = 1
    cfg.T2T_ViT.PARAMS.t2t_module_transformer_encoder_dropout = 0.0
    cfg.T2T_ViT.PARAMS.t2t_module_transformer_attention_dropout = 0.0
    cfg.T2T_ViT.PARAMS.t2t_module_transformer_projection_dropout = 0.0
    cfg.T2T_ViT.PARAMS.t2t_module_patch_size = 3
    cfg.T2T_ViT.PARAMS.t2t_module_overlapping = 1
    cfg.T2T_ViT.PARAMS.t2t_module_padding = 1
    cfg.T2T_ViT.PARAMS.soft_split_kernel_size = 7
    cfg.T2T_ViT.PARAMS.soft_split_stride = 4
    cfg.T2T_ViT.PARAMS.soft_split_padding = 2
    cfg.T2T_ViT.PARAMS.vit_backbone_d_ff = 1152
    cfg.T2T_ViT.PARAMS.vit_backbone_num_heads = 12
    cfg.T2T_ViT.PARAMS.vit_backbone_num_blocks = 14
    cfg.T2T_ViT.PARAMS.vit_backbone_encoder_dropout = 0.0
    cfg.T2T_ViT.PARAMS.vit_backbone_attention_dropout = 0.0
    cfg.T2T_ViT.PARAMS.vit_backbone_projection_dropout = 0.0
    cfg.T2T_ViT.PARAMS.vit_backbone_stodepth = True
    cfg.T2T_ViT.PARAMS.vit_backbone_stodepth_mp = 0.1
    cfg.T2T_ViT.PARAMS.vit_backbone_layer_scale = 1e-6
    cfg.T2T_ViT.PARAMS.vit_backbone_num_patches = 196
    cfg.T2T_ViT.PARAMS.vit_backbone_se_points = "msa"
    cfg.T2T_ViT.PARAMS.vit_backbone_qkv_bias = False
    cfg.T2T_ViT.PARAMS.vit_backbone_in_dims = None
    cfg.T2T_ViT.PARAMS.classifier_hidden_dim = 2048
    cfg.T2T_ViT.PARAMS.classifier_dropout = 0.0
    cfg.T2T_ViT.PARAMS.classifier_num_classes = 1000
    cfg.T2T_ViT.PARAMS.in_channels = 3
    cfg.T2T_ViT.PARAMS.image_size = 224

    cfg.TRAIN = CN() # set this
    cfg.TRAIN.PARAMS = CN()
    cfg.TRAIN.PARAMS.epochs = 1000
    cfg.TRAIN.PARAMS.gradient_accumulation = True
    cfg.TRAIN.PARAMS.gradient_accumulation_batch_size = 512
    cfg.TRAIN.PARAMS.gradient_clipping = ("norm", 1)
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
    cfg.DATA_MIXING.mixup_alpha = 0.8
    cfg.DATA_MIXING.cutmix_alpha = 1.0
    cfg.DATA_MIXING.cutmix_minmax = None
    cfg.DATA_MIXING.prob = 0.7
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
        dict(func="resizeWithAspectRatio", params=dict(size=224 if cfg.T2T_ViT.PARAMS.image_size == 192 else 256))
    ]

    cfg.T2T_ViT.DATALOADER_TRAIN_PARAMS = CN()
    cfg.T2T_ViT.DATALOADER_TRAIN_PARAMS.batch_size = 16
    cfg.T2T_ViT.DATALOADER_TRAIN_PARAMS.shuffle = True
    cfg.T2T_ViT.DATALOADER_TRAIN_PARAMS.num_workers = 8
    cfg.T2T_ViT.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.T2T_ViT.DATALOADER_TRAIN_PARAMS.drop_last = True

    cfg.T2T_ViT.DATALOADER_VAL_PARAMS = CN()
    cfg.T2T_ViT.DATALOADER_VAL_PARAMS.batch_size = 64
    cfg.T2T_ViT.DATALOADER_VAL_PARAMS.shuffle = True
    cfg.T2T_ViT.DATALOADER_VAL_PARAMS.num_workers = 8
    cfg.T2T_ViT.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.T2T_ViT.DATALOADER_VAL_PARAMS.drop_last = True

    cfg.T2T_ViT.TRANSFORMS = CN()

    cfg.DeiT.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomResizedCrop", params=dict(size=(cfg.T2T_ViT.PARAMS.image_size, cfg.T2T_ViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="RandomAugmentation", params=dict(m=9, n=2, mstd=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
        dict(name="RandomCutOut", params=dict(probability=0.25, mode="pixel", device="cpu"))
    ]

    cfg.T2T_ViT.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="CenterCrop", params=dict(size=(cfg.T2T_ViT.PARAMS.image_size, cfg.T2T_ViT.PARAMS.image_size))),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.T2T_ViT.OPTIMIZER = CN()
    cfg.T2T_ViT.OPTIMIZER.NAME = "AdamW"
    cfg.T2T_ViT.OPTIMIZER.PARAMS = CN()
    cfg.T2T_ViT.OPTIMIZER.PARAMS.lr = 5e-4 * cfg.num_gpus 
    cfg.T2T_ViT.OPTIMIZER.PARAMS.betas = (0.9, 0.999)
    cfg.T2T_ViT.OPTIMIZER.PARAMS.weight_decay = 5e-2

    cfg.T2T_ViT.LR_SCHEDULER = CN()
    cfg.T2T_ViT.LR_SCHEDULER.NAME = "WarmUpCosineLRScheduler"
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS = CN()
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.lr_initial = 1e-8
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.lr_final = cfg.T2T_ViT.OPTIMIZER.PARAMS.lr
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.warmup_epochs = 5
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.warmup_method = "linear"
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.after_scheduler = CN()
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.after_scheduler.NAME = "CosineAnnealingWarmRestarts"
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS = CN()
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_0 = 15
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.T_mult = 3
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.eta_min = 1e-8
    cfg.T2T_ViT.LR_SCHEDULER.PARAMS.after_scheduler.PARAMS.last_epoch = -1
    
    cfg.T2T_ViT.LOSS = CN()
    cfg.T2T_ViT.LOSS.NAME = "CrossEntropyLoss"
    cfg.T2T_ViT.LOSS.PARAMS = CN()
    cfg.T2T_ViT.LOSS.PARAMS.weight = None
    cfg.T2T_ViT.LOSS.PARAMS.size_average = None
    cfg.T2T_ViT.LOSS.PARAMS.ignore_index = -100
    cfg.T2T_ViT.LOSS.PARAMS.reduce = None
    cfg.T2T_ViT.LOSS.PARAMS.reduction = "mean"
    cfg.T2T_ViT.LOSS.PARAMS.label_smoothing = 0.1

    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0
