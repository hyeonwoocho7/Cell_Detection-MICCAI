import random
from pathlib import Path
from skimage.feature import peak_local_max
import numpy as np
import torch
import cv2
from scipy.ndimage.interpolation import rotate

def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(
        img, (pad_size, pad_size), "constant"
    )  # zero padding
    img_t = cv2.GaussianBlur(
        img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma
    )  # gaussian filter
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t

#픽셀값이 255가 되는 곳의 중심좌표를 반환한다.
def local_maxima(img, threshold=100, dist=2):
    #assert len(img.shape) == 2
    data = np.zeros((0, 2))
    #픽셀이 최대가 되는 좌표를 구한다.
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    #128x128행렬 만들기
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    #peak_img를 만들어서 픽셀이 최대가 되는 좌표에 255픽셀값을을 넣는다.
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    #labeling
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)
    return data

def make_pgt(img, threshold=200, dist=2):
    data = np.zeros((0, 2))
    #픽셀이 최대가 되는 좌표를 구한다.
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    #128x128행렬 만들기
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    #peak_img를 만들어서 픽셀이 최대가 되는 좌표에 255픽셀값을을 넣는다.
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    #labeling
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)

    black = np.zeros((128, 128))
    # likelihood map of one input
    result = black.copy()
    points = list(data)
    for point in points:
        img_t = black.copy()  # likelihood map of one cell
        img_t[int(point[1])][int(point[0])] = 255  # plot a white dot
        img_t = gaus_filter(img_t, 101, 6)
        result = np.maximum(result, img_t)  # compare result with gaussian_img
    #  normalization
    if result.max() == 0:
        result = result
    else:
        result = 255 * result / result.max()
    result = result.astype("uint8")
    return result

class CellImageLoad1(object):
    def __init__(self, ori_path, gt_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path



    def __len__(self):
        return len(self.ori_paths)

    def change_brightness(self, image, value):
        """

        Args:

            image : numpy array of image

            value : brightness

        Return :

            image : numpy array of image with brightness added

        """

        # image = image.astype("int16")

        image = image + value

        # image = ceil_floor_image(image)

        return image

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = cv2.imread(str(img_name), 0)
        img = img / 255

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gt = gt / 255

        img, gt = self.data_augment(img, gt)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0)}

        return datas

    def data_augment(self, img, gt):
        rand_value = np.random.randint(0, 4)
        img = rotate(img, 90 * rand_value, mode="nearest")
        gt = rotate(gt, 90 * rand_value)

        # pix_add = random.uniform(-0.1, 0.1)
        # img = self.change_brightness(img, pix_add)
        #
        # img = (img-img.min()) / (1 + pix_add -img.min())

        return img, gt




class CellImageLoad2(object):
    def __init__(self, ori_path, gt_path, crop_size=(256, 256)):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        #self.crop_size = crop_size
        self.img_list = []
        self.label_list = []
        self.gt_list = []
        self.B2_num = 0
        self.D4_num = 0
        self.seq2_num = 0
        self.makelabel()

    def makelabel(self):
        for ori_path, gt_path in zip(self.ori_paths, self.gt_paths):
            if 'B2_1' in Path(ori_path).stem:
                self.B2_num += 1
                self.label_list.append(0)
            elif 'D4_1' in Path(ori_path).stem:
                self.D4_num += 1
                self.label_list.append(1)
            else:
                self.seq2_num += 1
                self.label_list.append(2)

            img = cv2.imread(str(ori_path), 0)
            img = img / 255
            rand_value = np.random.randint(0, 4)
            img = rotate(img, 90 * rand_value, mode="nearest")
            img = torch.from_numpy(img.astype(np.float32))
            self.img_list.append(img.unsqueeze(0))
            gt = cv2.imread(str(gt_path), 0)
            gt = gt / 255
            gt = rotate(gt, 90 * rand_value, mode="nearest")
            gt = torch.from_numpy(gt.astype(np.float32))
            self.gt_list.append(gt.unsqueeze(0))


    def __len__(self):
        #return len(self.ori_paths)
        return len(self.img_list)
    '''
    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right
    '''
    def __getitem__(self, data_id):
        # img_name = self.ori_paths[data_id]
        # img = cv2.imread(str(img_name), 0)
        # img = img / 255
        #
        # gt_name = self.gt_paths[data_id]
        # gt = cv2.imread(str(gt_name), 0)
        # gt = gt / 255
        # '''
        # # data augumentation
        # top, bottom, left, right = self.random_crop_param(img.shape)
        #
        # img = img[top:bottom, left:right]
        # gt = gt[top:bottom, left:right]
        # '''
        # rand_value = np.random.randint(0, 4)
        # img = rotate(img, 90 * rand_value, mode="nearest")
        # gt = rotate(gt, 90 * rand_value)
        #
        # img = torch.from_numpy(img.astype(np.float32))
        # gt = torch.from_numpy(gt.astype(np.float32))

        #datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0)}
        datas = {"image": self.img_list[data_id], "gt": self.gt_list[data_id]}

        return datas

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
        img = img / 255

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