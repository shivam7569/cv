def Fast_RCNNConfig(cfg):
    from configs.config import CfgNode as CN

    cfg.LOGGING.NAME = "Fast_RCNN"
    cfg.TENSORBOARD.BASENAME = "Fast_RCNN"
    cfg.CHECKPOINT.BASENAME = "Fast_RCNN"
    cfg.CHECKPOINT.TREE = True
    cfg.CHECKPOINT.SAVE_EPOCH_CHECKPOINTS = True

    cfg.FAST_RCNN = CN()

    cfg.FAST_RCNN.BACKBONE = CN()
    cfg.FAST_RCNN.BACKBONE.NAME = "AlexNet"

    cfg.FAST_RCNN.RCNN_ROI_CSV_DIR = "detection/RCNN/data/finetune/"
    cfg.FAST_RCNN.ROI_CSV_DIR = "detection/Fast_RCNN/data/"
    cfg.FAST_RCNN.IMAGE_BATCH_SIZE = 2
    cfg.FAST_RCNN.IMAGE_POSITIVE_ROI_BATCH_SIZE = 16
    cfg.FAST_RCNN.IMAGE_NEGATIVE_ROI_BATCH_SIZE = 48

    cfg.PIPELINES = CN()
    cfg.PIPELINES.TRAIN = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="fast_rcnn_resize", params=dict(min_size=600, max_size=1000))
    ]
    cfg.PIPELINES.VAL = [
        dict(func="readImage", params=dict(uint8=True)),
        dict(func="fast_rcnn_resize", params=dict(min_size=600, max_size=1000))
    ]

    cfg.FAST_RCNN.TRANSFORMS = CN()
    cfg.FAST_RCNN.TRANSFORMS.TRAIN = [
        dict(name="DetectionHorizontalFlip", params=dict(p=0.5)),
        dict(name="ToPILImage", params=None),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]
    cfg.FAST_RCNN.TRANSFORMS.VAL = [
        dict(name="ToPILImage", params=None),
        dict(name="ToTensor", params=None),
        dict(name="Normalize", params=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    ]