import os
import numpy as np
from pathlib import Path
from utils import path_to_video
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class VideoDataset(Dataset):
    def __init__(self, directory, mode='train', clip_len=64, split=0.0, seed=1234, transform=None):
        folder = Path(directory)
        self.clip_len = clip_len
        self.transform = transform
        self.seed = seed
        self.mode = mode

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=np.int)

        if split != 0.0:
            x_train, x_valid, y_train, y_valid = train_test_split(self.fnames, self.label_array, test_size=split,
                                                                  random_state=self.seed)
            if mode == 'train':
                self.fnames = x_train
                self.label_array = y_train
            elif mode == 'valid':
                self.fnames = x_valid
                self.label_array = y_valid

        print(f'{mode} {self.__len__()} videos')

    def num_classes(self):
        return len(self.label2index)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, item):
        label = self.label_array[item]
        file_name = self.fnames[item]

        video = path_to_video(file_name, self.clip_len)

        if self.transform is not None:
            video = self.transform(video)

        return video, label
