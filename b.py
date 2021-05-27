import glob
import cv2
import os
import numpy as np
# paths = sorted(glob.glob("/home/hyeonwoo/research/D2_1/ori/*.tif"))
#
# for path in paths[:2700]:
#     img = cv2.imread("/home/hyeonwoo/research/D2_1/ori/"+os.path.basename(path), 0)
#     cv2.imwrite("/home/hyeonwoo/research/C2+D2/ori/"+os.path.basename(path), img)
ori_paths = sorted(glob.glob("/home/hyeonwoo/research/Experiment1/Result/Detection/step1/ori/*.tif"))
pgt_paths = sorted(glob.glob("/home/hyeonwoo/research/Experiment1/Result/Detection/step1/pgt/*.tif"))
for ori_path, pgt_path in zip(ori_paths, pgt_paths):
    ori = cv2.imread(ori_path, 0)
    pgt = cv2.imread(pgt_path, 0)
    imgs = []
    imgs.append(ori)
    imgs.append(pgt)
    img = np.hstack(imgs)
    cv2.imwrite("/home/hyeonwoo/research/Experiment1/Result/Detection/step1/k/"+os.path.basename(ori_path), img)
