import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image




class CellImage(object):
    def __init__(self, dir_paths):
        self.dir_paths = dir_paths

        self.datas = []

        for dir_path in dir_paths:
            origin_dir = Path(dir_path) / "origin"
            good_dir = Path(dir_path) / "good"
            bad_dir = Path(dir_path) / "bad"

            for img_path in origin_dir.iterdir():
                if img_path.is_file():
                    good_path = good_dir / img_path.name
                    bad_path = bad_dir / img_path.name

                    angle_type = np.random.randint(4, size=2*5)
                    flip_type = np.random.randint(3, size=2*5)
                    for i in range(5):
                        self.datas.append((img_path, good_path, 1, angle_type[i * 2], flip_type[i * 2]))
                        self.datas.append((img_path, bad_path, 0, angle_type[i * 2 + 1], flip_type[i * 2 + 1]))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, data_id):
        img_name, gt_name, label, angle, k = self.datas[data_id]

        img = cv2.imread(str(img_name), 0)
        img = np.asarray(img) / 255

        gt = cv2.imread(str(gt_name), 0)
        gt = gt / 255

        img, gt = augment(img, gt, angle, k)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0), "label": label}

        return datas


def augment(img, gt, angle, k):
    # data augumentation
    # rotate
    # flip
    # intensity
    img = np.rot90(img, k=angle)
    gt = np.rot90(gt, k=angle)

    if k == 0:
        img = np.flip(img)
        gt = np.flip(gt)
    if k == 1:
        img = np.flip(img, axis=0)
        gt = np.flip(gt, axis=0)
    if k == 2:
        img = np.flip(img, axis=1)
        gt = np.flip(gt, axis=1)

    return img, gt






