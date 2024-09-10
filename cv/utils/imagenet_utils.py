import os
import random
import xml.etree.ElementTree as ET

from cv.configs.config import get_cfg
from cv.utils import MetaWrapper

random.seed(420)

class ImagenetData(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class to parse and prepare ImageNet data"

    cfg = get_cfg()

    def __init__(self):
        self.cfg = get_cfg()

    def generateClassIdVsName(self):
        self.class_id_vs_name = {}
        with open(self.cfg.DATA.IMAGENET_CLASS_MAPPING, "r") as f:
            for line in f.readlines():
                class_id = line.strip().split(" ")[0]
                class_name = line.strip().split(" ")[1].split(",")[0].strip(" ")
                self.class_id_vs_name[class_id] = class_name

        with open(self.cfg.DATA.IMAGENET_CLASS_VS_NAME_TXT, "w") as f:
            for k, v in self.class_id_vs_name.items():
                f.write(f"{k} {v}\n")

    @classmethod
    def getClassVsName(cls):
        class_vs_name = {}
        
        try:
            with open(cls.cfg.DATA.IMAGENET_CLASS_VS_NAME_TXT, "r") as f:
                for line in f.readlines():
                    class_id = line.strip().split(" ")[0]
                    class_name = line.strip().split(" ")[1]
                    class_vs_name[class_id] = class_name
        except:
            from cv.datasets.classification import imagenet_txts
            import importlib.resources as pkg_resources

            with pkg_resources.open_text(imagenet_txts, 'class_vs_name.txt') as f:
                for line in f.readlines():
                    class_id = line.strip().split(" ")[0]
                    class_name = line.strip().split(" ")[1]
                    class_vs_name[class_id] = class_name

        return class_vs_name
    
    @classmethod
    def getIdVsName(cls):
        id_vs_name = {}

        try:
            with open(cls.cfg.DATA.IMAGENET_CLASS_VS_ID_TXT, "r") as f:
                for line in f.readlines():
                    class_id, class_int = line.strip().split(" ")
                    class_name = cls.getClassVsName()[class_id]
                    id_vs_name[class_int] = class_name
        except:
            from cv.datasets.classification import imagenet_txts
            import importlib.resources as pkg_resources

            with pkg_resources.open_text(imagenet_txts, 'class_vs_id.txt') as f:
                for line in f.readlines():
                    class_id, class_int = line.strip().split(" ")
                    class_name = cls.getClassVsName()[class_id]
                    id_vs_name[class_int] = class_name

        return id_vs_name

    def segregateData(self):
        self.getImageNetClasses()
        self.generateImagesTextFiles()
        self.writeClassVsId()

    def writeClassVsId(self):
        with open(self.cfg.DATA.IMAGENET_CLASS_VS_ID_TXT, "w") as f:
            for k, v in self.class_vs_id.items():
                f.write(f"{k} {v}\n")

    def getImageNetClasses(self):

        self.num_class_images = {}
        with open(self.cfg.DATA.IMAGENET_CLASS_MAPPING, "r") as f:
            for line in f.readlines():
                class_id = line.strip().split(" ")[0]
                self.num_class_images[class_id] = len(
                    os.listdir(
                        os.path.join(self.cfg.DATA.IMAGENET_TRAIN_IMAGES, class_id)
                    )
                )

        self.num_class_images = dict(sorted(self.num_class_images.items(), key=lambda x: x[1], reverse=True))
        self.imagenet_classes = list(self.num_class_images.keys())
        self.class_vs_id = {class_id: i for i, class_id in enumerate(self.imagenet_classes)}

    def generateImagesTextFiles(self):
        train_images = []
        for class_id in self.imagenet_classes:
            class_imgs_path = os.path.join(self.cfg.DATA.IMAGENET_TRAIN_IMAGES, class_id)
            train_images.extend([
                os.path.join(class_imgs_path, i) + " " + str(self.class_vs_id[class_id]) 
                for i in os.listdir(class_imgs_path)
                ])

        with open(self.cfg.DATA.IMAGENET_TRAIN_TXT, "w") as f:
            for line in train_images:
                f.write(line+"\n")

        val_images = []
        val_annotations = os.listdir(self.cfg.DATA.IMAGENET_VAL_ANNOTATIONS)
        for val_annot in val_annotations:
            val_data = ImagenetData.parseValAnnotation(os.path.join(self.cfg.DATA.IMAGENET_VAL_ANNOTATIONS, val_annot))

            if val_data[1] in self.imagenet_classes:
                val_images.append(
                    os.path.join(self.cfg.DATA.IMAGENET_VAL_IMAGES, val_data[0]) + ".JPEG" + " " + str(self.class_vs_id[val_data[1]])
                )

        with open(self.cfg.DATA.IMAGENET_VAL_TXT, "w") as f:
            for line in val_images:
                f.write(line+"\n")
        
    @classmethod
    def parseValAnnotation(cls, annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        filename = root.findall("filename")[0].text

        object_ = root.findall("object")
        class_id = [i.findall("name") for i in object_][0][0].text

        return (filename, class_id)
