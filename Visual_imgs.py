import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
# Micaii
path = "B2+B4=D4/200/3"
def vis(path, step):
    ori_paths = sorted(glob.glob("/home/hyeonwoo/research/Micaii/{}/All_Result/Detection/step{}/ori/*.tif".format(str(path), str(step))))
    gt_paths = sorted(glob.glob("/home/hyeonwoo/research/Micaii/{}/All_Result/Detection/step{}/gt/*.tif".format(str(path), str(step))))
    pred_paths = sorted(glob.glob("/home/hyeonwoo/research/Micaii/{}/All_Result/Detection/step{}/pred/*.tif".format(str(path), str(step))))
    #pgt_paths = sorted(glob.glob("/home/hyeonwoo/research/Micaii/{}/All_Result/Detection/step{}/pgt/*.tif".format(str(path), str(step))))

    os.makedirs("/home/hyeonwoo/research/Micaii/{}/All_Result/Detection/step{}/Result/".format(str(path), str(step)), exist_ok=True)
    for ori_path, gt_path, pred_path in zip(ori_paths, gt_paths,pred_paths):
        imgs = []
        ori = cv2.imread(ori_path, 0)
        gt = cv2.imread(gt_path, 0)
        pred = cv2.imread(pred_path, 0)
        #pred = cv2.imread(str("/home/hyeonwoo/research/Micaii/{}/All_Result/Detection/step{}/pred/".format(str(path), str(step))+os.path.basename(ori_path)), 0)
        #pgt = cv2.imread(pgt_path, 0)
        imgs.append(ori)
        imgs.append(gt)
        imgs.append(pred)
        #imgs.append(pgt)
        img = np.hstack(imgs)
        cv2.imwrite("/home/hyeonwoo/research/Micaii/{}/All_Result/Detection/step{}/Result/".format(str(path), str(step))+os.path.basename(ori_path), img)

vis(path, 0)
vis(path, 3)






