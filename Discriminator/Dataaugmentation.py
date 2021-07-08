from pathlib import Path
import numpy as np
from numpy import linalg as LA
import cv2
from .utils import local_maxima, set_seed
import math
import random


def gaussian(x, sigma, mu):
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    n = x.ndim
    return np.exp(-np.diag((x - mu) @ inv @ (x - mu).T) / 2.0) / (np.sqrt((2 * np.pi) ** n * det))


def tangent_angle(u: np.ndarray, v: np.ndarray):
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    return np.rad2deg(np.arccos(c))

def data_augmentation(train_path,  save_train_ori, save_train_good, save_train_bad, step):
    train_ori_path = train_path / Path("ori/")
    train_gt_path = train_path / Path("gt/")
    train_ori_paths = sorted(train_ori_path.glob("*.tif"))
    train_gt_paths = sorted(train_gt_path.glob("*.tif"))
    generation_train(train_ori_paths, train_gt_paths, save_train_ori, save_train_good, save_train_bad, step)


def generation_train(ori_paths, gt_paths, save_train_ori, save_train_good, save_train_bad, step):
    if step == 0:
        m = 5
    if step > 0:
        m = 3
    for times in range(0, m):
        frame = 1 + len(ori_paths) * times
        for ori_path, gt_path in zip(ori_paths, gt_paths):
            img = cv2.imread(str(ori_path), 0)
            cv2.imwrite(str(save_train_ori.joinpath("{:05d}.tif".format(frame))), img)
            mask = cv2.imread(str(gt_path), 0)
            gaus = np.zeros_like(mask, dtype=np.float)
            center = local_maxima(mask, threshold=200, dist=2)
            # id list
            k = list(range(0, len(center)))
            random_id = sorted(random.sample(k, random.randint(0, len(k))))
            # sum of distance
            Dis = 0

            for id, location in enumerate(center[:]):

                sigma = np.array([[36, 0],
                                  [0, 36]])

                x_pos = int(location[0])
                y_pos = int(location[1])
                x_shift = random.randint(-1, 1)
                y_shift = random.randint(-1, 1)
                x_pos = int(location[0]) + x_shift
                x_pos = max(min(x_pos, mask.shape[0]), 0)
                y_pos = int(location[1]) + y_shift
                y_pos = max(min(y_pos, mask.shape[1]), 0)

                dist = math.sqrt((x_shift * x_shift) + (y_shift * y_shift))

                gau = np.zeros((mask.shape[0] + 101, mask.shape[1] + 101))
                X, Y = np.meshgrid(np.arange(101), np.arange(101))
                if len(center) >= 2:
                    if id in random_id:
                        gau_local = gaussian(np.c_[X.flatten(), Y.flatten()], sigma, mu=np.array([50, 50]))
                        gau_local = gau_local.reshape((101, 101))
                        gau_local = 255 * gau_local / gau_local.max()
                    else:
                        gau_local = gaussian(np.c_[X.flatten(), Y.flatten()], sigma, mu=np.array([50, 50]))
                        gau_local = gau_local.reshape((101, 101))
                        gau_local = 255 * gau_local / gau_local.max()
                else:
                    gau_local = gaussian(np.c_[X.flatten(), Y.flatten()], sigma, mu=np.array([50, 50]))
                    gau_local = gau_local.reshape((101, 101))
                    gau_local = 255 * gau_local / gau_local.max()
                gau[y_pos:y_pos + 101, x_pos:x_pos + 101] = gau_local
                gaus = np.maximum(gaus, gau[50: -51, 50: -51])
                gaus = gaus.astype(np.uint8)

            cv2.imwrite(
                str(save_train_good.joinpath("{:05d}.tif".format(frame))),
                gaus.astype(np.uint8))
            gaus = np.zeros_like(mask, dtype=np.float)
            center = local_maxima(mask, threshold=200, dist=2)
            del_center = center
            erazed_center = np.zeros((0, 2))
            # random select the number of cell erazed
            n = random.choice(list(range(0, 3)))
            # erazing radom centers
            if n <= len(center):
                for i in range(0, n):
                    raw = random.choice(list(range(0, len(del_center))))
                    erazed_center = np.append(erazed_center, [del_center[raw, :]], axis=0)
                    del_center = np.delete(del_center, raw, 0)
            if 0 < len(center) < n:
                n = random.choice(list(range(0, len(center))))
                for ii in range(0, len(center)):
                    raw = random.choice(list(range(0, len(del_center))))
                    erazed_center = np.append(erazed_center, [del_center[raw, :]], axis=0)
                    del_center = np.delete(del_center, raw, 0)
            # the number of cell appended
            if n == 0 or len(center) == 0:
                p = random.choice([1, 2])
            if n != 0:
                p = random.choice([0, 1, 2])
            plus_center = del_center
            for j in range(0, p):
                x = random.choice(list(range(15, 110)))
                y = random.choice(list(range(15, 110)))
                a = np.array((x, y))
                count = 0
                if not [x, y] in erazed_center:
                    for location in (center[:]):
                        dis = np.linalg.norm(a - location)
                        if dis > 15:
                            count += 1
                    if count == len(center):
                        plus_center = (np.append(plus_center, [[x, y]], axis=0))
            new_center = plus_center
            if np.array_equal(new_center, center):
                cv2.imwrite(
                    str(save_train_bad.joinpath("{:05d}.tif".format(frame))),
                    gaus.astype(np.uint8))

            else:
                for id, loc in enumerate(new_center[:]):
                    x_pos = int(loc[0])
                    y_pos = int(loc[1])
                    sigma = np.array([[36, 0],
                                      [0, 36]])
                    gau = np.zeros((mask.shape[0] + 101, mask.shape[1] + 101))
                    X, Y = np.meshgrid(np.arange(101), np.arange(101))
                    gau_local = gaussian(np.c_[X.flatten(), Y.flatten()], sigma, mu=np.array([50, 50]))
                    gau_local = gau_local.reshape((101, 101))
                    gau_local = 255 * gau_local / gau_local.max()
                    gau[y_pos:y_pos + 101, x_pos:x_pos + 101] = gau_local
                    gaus = np.maximum(gaus, gau[50: -51, 50: -51])
                    gaus = gaus.astype(np.uint8)
                cv2.imwrite(
                    str(save_train_bad.joinpath("{:05d}.tif".format(frame))),
                    gaus.astype(np.uint8))
            frame += 1

def Dataaugmentation(step, seed, save, Det_Data_path):
    set_seed(seed)

    save_train = save
    save_train_ori = save_train / Path("origin")
    save_train_good = save_train / Path("good")
    save_train_bad = save_train / Path("bad")


    save_train_ori.mkdir(parents=True, exist_ok=True)
    save_train_good.mkdir(parents=True, exist_ok=True)
    save_train_bad.mkdir(parents=True, exist_ok=True)

    data_augmentation(Det_Data_path, save_train_ori, save_train_good, save_train_bad, step)