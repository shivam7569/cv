import os
import cv2
import pandas as pd
from tqdm import tqdm
from cv.utils import MetaWrapper
from cv.configs.config import get_cfg
from cv.utils.os_utils import check_dir


class MNISTData(metaclass=MetaWrapper):

    @classmethod
    def __class_repr__(cls):
        return "Class to parse and prepare MNIST data"
    
    cfg = get_cfg()
    
    @classmethod
    def csvToImages(cls):
        data_dir = os.path.dirname(cls.cfg.DATA.MNIST_TRAIN_CSV)
        train_data_dir = os.path.join(data_dir, "train")
        test_data_dir = os.path.join(data_dir, "test")

        check_dir(path=train_data_dir, create=True, forcedCreate=False, tree=False)
        check_dir(path=test_data_dir, create=True, forcedCreate=False, tree=False)

        # training data

        train_df = pd.read_csv(cls.cfg.DATA.MNIST_TRAIN_CSV)
        num_images = train_df.shape[0]

        for i in tqdm(range(num_images)):
            img_row = train_df.iloc[i, 1:].to_numpy()
            label = train_df.iloc[i, 0]
            img = img_row.reshape(28, 28)
            img_name = f"mnist_train_image_{str(i).zfill(len(str(num_images)))}_class_{label}.png"
            img_path = os.path.join(train_data_dir, img_name)
            cv2.imwrite(img_path, img)

        # validation data
        
        test_df = pd.read_csv(cls.cfg.DATA.MNIST_TEST_CSV)
        num_images = test_df.shape[0]

        for i in tqdm(range(num_images)):
            img_row = test_df.iloc[i, 1:].to_numpy()
            label = test_df.iloc[i, 0]
            img = img_row.reshape(28, 28)
            img_name = f"mnist_test_image_{str(i).zfill(len(str(num_images)))}_class_{label}.png"
            img_path = os.path.join(test_data_dir, img_name)
            cv2.imwrite(img_path, img)

    @classmethod
    def prepareTxtFiles(cls):

        # training txt files

        train_files = [
            os.path.join(
                os.path.join(
                    os.path.dirname(
                        cls.cfg.DATA.MNIST_TRAIN_CSV
                    ), "train"
                ), i
            ) for i in os.listdir(
                os.path.join(
                    os.path.dirname(
                        cls.cfg.DATA.MNIST_TRAIN_CSV
                    ), "train"
                )
            )
        ]

        train_files_with_labels = [
            ' '.join((i, os.path.splitext(os.path.basename(i))[0].split('_')[-1])) for i in train_files
        ]

        with open(cls.cfg.DATA.MNIST_TRAIN_TXT, 'w') as f:
            for line in train_files_with_labels:
                f.write(line+'\n')

        # validation text file
                
        val_files = [
            os.path.join(
                os.path.join(
                    os.path.dirname(
                        cls.cfg.DATA.MNIST_TEST_CSV
                    ), "test"
                ), i
            ) for i in os.listdir(
                os.path.join(
                    os.path.dirname(
                        cls.cfg.DATA.MNIST_TEST_CSV
                    ), "test"
                )
            )
        ]

        val_files_with_labels = [
            ' '.join((i, os.path.splitext(os.path.basename(i))[0].split('_')[-1])) for i in val_files
        ]

        with open(cls.cfg.DATA.MNIST_TEST_TXT, 'w') as f:
            for line in val_files_with_labels:
                f.write(line+'\n')

    @staticmethod
    def setupMNISTData():
        MNISTData.csvToImages()
        MNISTData.prepareTxtFiles()