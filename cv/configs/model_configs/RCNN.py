def RCNNConfig(cfg):
    from cv.configs.config import CfgNode as CN

    cfg.LOGGING.NAME = "RCNN"
    cfg.TENSORBOARD.BASENAME = "RCNN/Finetuning"
    cfg.CHECKPOINT.BASENAME = "RCNN/Finetuning"
    cfg.CHECKPOINT.TREE = True
    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True
    
    cfg.RCNN = CN()

    cfg.RCNN.SelectiveSearch = CN()
    cfg.RCNN.SelectiveSearch.STRATEGY = 'f'
    cfg.RCNN.SelectiveSearch.AREA_THRESHOLD = 10000
    cfg.RCNN.SelectiveSearch.NUM_PROPOSALS = 2000

    cfg.RCNN.Finetune = CN()
    cfg.RCNN.Finetune.BACKBONE = "AlexNet"
    cfg.RCNN.Finetune.ROI_THRESHOLD = 0.5
    cfg.RCNN.Finetune.BATCH_POSITIVES = 32
    cfg.RCNN.Finetune.BATCH_NEGATIVES = 96
    cfg.RCNN.Finetune.DATA_DIR = "detection/RCNN/data/finetune/"
    cfg.RCNN.Finetune.TRANSFORMS = CN()
    cfg.RCNN.Finetune.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomCrop", params=dict(size=(224, 224), pad_if_needed=False)),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.RCNN.Finetune.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.RCNN.Finetune.OPTIMIZER = CN()
    cfg.RCNN.Finetune.OPTIMIZER.NAME = "SGD"
    cfg.RCNN.Finetune.OPTIMIZER.PARAMS = CN()
    cfg.RCNN.Finetune.OPTIMIZER.PARAMS.lr = 0.001
    cfg.RCNN.Finetune.OPTIMIZER.PARAMS.momentum = 0.9
    cfg.RCNN.Finetune.OPTIMIZER.PARAMS.weight_decay = 0.0

    cfg.RCNN.Finetune.LR_SCHEDULER = CN()
    cfg.RCNN.Finetune.LR_SCHEDULER.NAME = "StepLR"
    cfg.RCNN.Finetune.LR_SCHEDULER.PARAMS = CN()
    cfg.RCNN.Finetune.LR_SCHEDULER.PARAMS.step_size = 5
    cfg.RCNN.Finetune.LR_SCHEDULER.PARAMS.gamma = 0.1
    cfg.RCNN.Finetune.LR_SCHEDULER.PARAMS.verbose = False

    cfg.RCNN.Finetune.LOSS = CN()
    cfg.RCNN.Finetune.LOSS.NAME = "CrossEntropyLoss"
    cfg.RCNN.Finetune.LOSS.PARAMS = CN()
    cfg.RCNN.Finetune.LOSS.PARAMS.weight = None
    cfg.RCNN.Finetune.LOSS.PARAMS.size_average = None
    cfg.RCNN.Finetune.LOSS.PARAMS.ignore_index = -100
    cfg.RCNN.Finetune.LOSS.PARAMS.reduce = None
    cfg.RCNN.Finetune.LOSS.PARAMS.reduction = "mean"

    cfg.RCNN.Classifier = CN()
    cfg.RCNN.Classifier.DATA_DIR = "detection/RCNN/data/classification/"
    cfg.RCNN.Classifier.ROI_THRESHOLD = 0.3
    cfg.RCNN.Classifier.TRANSFORMS = CN()
    cfg.RCNN.Classifier.TRANSFORMS.TRAIN = [
        dict(name="ToPILImage", params=None),
        dict(name="RandomHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.RCNN.Classifier.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.Finetune = CN()
    cfg.PIPELINES.Finetune.TRAIN = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.Finetune.VAL = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.Finetune.ROI = [
        dict(func="extractROI", params=dict(dilated=0)),
        dict(func="rcnn_warpROI", params=dict(size=(224, 224)))
    ]
    cfg.PIPELINES.Classifier = CN()
    cfg.PIPELINES.Classifier.TRAIN = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.Classifier.VAL = [
        dict(func="readImage", params=dict(uint8=True))
    ]
    cfg.PIPELINES.Classifier.ROI = [
        dict(func="extractROI", params=dict(dilated=16)),
        dict(func="rcnn_warpROI", params=dict(size=(224, 224)))
    ]

    cfg.RCNN.DATALOADER_TRAIN_PARAMS = CN()
    cfg.RCNN.DATALOADER_TRAIN_PARAMS.batch_size = 128
    cfg.RCNN.DATALOADER_TRAIN_PARAMS.num_workers = 16
    cfg.RCNN.DATALOADER_TRAIN_PARAMS.pin_memory = True
    cfg.RCNN.DATALOADER_TRAIN_PARAMS.drop_last = True
    
    cfg.RCNN.DATALOADER_VAL_PARAMS = CN()
    cfg.RCNN.DATALOADER_VAL_PARAMS.batch_size = 128
    cfg.RCNN.DATALOADER_VAL_PARAMS.num_workers = 16
    cfg.RCNN.DATALOADER_VAL_PARAMS.pin_memory = True
    cfg.RCNN.DATALOADER_VAL_PARAMS.drop_last = True

    assert cfg.RCNN.DATALOADER_TRAIN_PARAMS.batch_size == cfg.RCNN.Finetune.BATCH_POSITIVES + cfg.RCNN.Finetune.BATCH_NEGATIVES
    assert cfg.RCNN.DATALOADER_VAL_PARAMS.batch_size == cfg.RCNN.Finetune.BATCH_POSITIVES + cfg.RCNN.Finetune.BATCH_NEGATIVES
    
    cfg.REGULARIZATION = CN()
    cfg.REGULARIZATION.MODE = ''
    cfg.REGULARIZATION.STRENGTH = 0