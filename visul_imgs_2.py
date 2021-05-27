import cv2
import numpy as np
import glob
import os
path = "con_to_BMP2/Experiment"
def vis(path, step):
    ori_paths = sorted(glob.glob("/home/hyeonwoo/research/semi/{}/Result/Discriminator/step{}/save_selected_top/ori/*.tif".format(str(path), str(step))))
    gt_paths = sorted(glob.glob("/home/hyeonwoo/research/semi/{}/Result/Discriminator/step{}/save_selected_top/gt/*.tif".format(str(path), str(step))))
    #pred_paths = sorted(glob.glob("/home/hyeonwoo/research/semi/{}/Result/Discriminator/step{}/pred/*.tif".format(str(path), str(step))))
    pgt_paths = sorted(glob.glob("/home/hyeonwoo/research/semi/{}/Result/Discriminator/step{}/save_selected_top/pgt/*.tif".format(str(path), str(step))))

    os.makedirs("/home/hyeonwoo/research/semi/{}/Result/Discriminator/step{}/Result/".format(str(path), str(step)), exist_ok=True)
    for ori_path, gt_path, pgt_path in zip(ori_paths, gt_paths, pgt_paths):
        imgs = []
        ori = cv2.imread(ori_path, 0)
        gt = cv2.imread(gt_path, 0)
        #pred = cv2.imread(pred_path, 0)
        pred = cv2.imread(str("/home/hyeonwoo/research/semi/{}/Result/Detection/step{}/pred/".format(str(path), str(step))+os.path.basename(ori_path)), 0)
        pgt = cv2.imread(pgt_path, 0)
        imgs.append(ori)
        imgs.append(gt)
        imgs.append(pred)
        imgs.append(pgt)
        img = np.hstack(imgs)
        cv2.imwrite("/home/hyeonwoo/research/semi/{}/Result/Discriminator/step{}/Result/".format(str(path), str(step))+os.path.basename(ori_path), img)

for i in range(4):
    vis(path, i)