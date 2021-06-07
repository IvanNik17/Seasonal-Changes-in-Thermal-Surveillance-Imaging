import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import pandas as pd

import torch.utils.data as data
import torchvision.transforms as transforms

rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, split, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.split = split
        self.dir = video_folder
        self.image_dir = '/home/jacob/data/habor/image_dataset/'
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.samples = self.get_all_habor_samples()


    def get_all_habor_samples(self, split_ratio=0.8):
        df = pd.read_csv(self.dir)

        # use 80% of train data for validation
        mask = np.random.rand(len(df)) < split_ratio
        if self.split == 'train':
            df_train = df[mask]
            return df_train
        elif self.split == 'validation':
            df_val = df[~mask]
            return df_val
        else:
            return df


    def __getitem__(self, index):
        sample = self.samples.iloc[index]

        batch = []
        for i in reversed(range(self._time_step+self._num_pred)):
            image_number = sample['Image Number']
            image_number = f"image_{int(image_number) - i :04}"
            image_path = os.path.join(self.image_dir,
                                      str(sample['Folder name']),
                                      sample['Clip Name'],
                                      image_number + '.jpg')

            image = np_load_frame(image_path, self._resize_height, self._resize_width)

            if self.transform is not None:
                batch.append(self.transform(image))
        return (np.concatenate(batch, axis=0), image_path)


    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train_folder = r'/home/jacob/code/Seasonal-Changes-in-Thermal-Surveillance-Imaging/splits/coldest_day_5000.csv'

    # recon
    t_length = 1
    train_dataset_recon = DataLoader('train', train_folder, transforms.Compose([
                 transforms.ToTensor(),          
                 ]), resize_height=256, resize_width=256, time_step=t_length-1)
    sample0 = train_dataset_recon[0]

    # pred
    train_folder = '/home/jacob/data/habor/image_dataset/datasplit/coldest_day.csv'
    t_length = 5
    train_dataset_pred = DataLoader('train', train_folder, transforms.Compose([
                 transforms.ToTensor(),          
                 ]), resize_height=256, resize_width=256, time_step=t_length-1)
    sample1 = train_dataset_pred[0]

    __import__('ipdb').set_trace()
