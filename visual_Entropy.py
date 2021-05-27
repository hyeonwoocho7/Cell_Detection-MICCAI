import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt



path = "Experiment1"
def vis(path, step):
    ori_paths = sorted(glob.glob("/home/hyeonwoo/research/{}/Result/Detection/step{}/ori/*.tif".format(str(path), str(step))))
    gt_paths = sorted(glob.glob("/home/hyeonwoo/research/{}/Result/Detection/step{}/gt/*.tif".format(str(path), str(step))))
    pred_paths = sorted(glob.glob("/home/hyeonwoo/research/{}/Result/Detection/step{}/pred/*.tif".format(str(path), str(step))))
    pgt_paths = sorted(glob.glob("/home/hyeonwoo/research/{}/Result/Detection/step{}/pgt/*.tif".format(str(path), str(step))))
    Entropy_txt = open("/home/hyeonwoo/research/{}/Result/Discriminator/step{}/Entropy.txt".format(str(path), str(step)))
    data = Entropy_txt.read()
    datas = data.split("\n")
    datas = datas[:-1]
    Entropy = []
    probability = []
    for i, data in enumerate(datas):
        data = data.split(', ')
        Entropy.append(data[0])
        probability.append(data[1])
    os.makedirs("/home/hyeonwoo/research/{}/Result/Discriminator/step{}/Result2/".format(str(path), str(step)), exist_ok=True)
    j = 0
    for ori_path, gt_path, pred_path,  pgt_path in zip(ori_paths, gt_paths,pred_paths, pgt_paths):
        imgs = []
        ori = cv2.imread(ori_path, 0)
        gt = cv2.imread(gt_path, 0)
        pred = cv2.imread(pred_path, 0)
        #pred = cv2.imread(str("/home/hyeonwoo/research/{}/Result/Detection/step{}/pred/".format(str(path), str(step))+os.path.basename(ori_path)), 0)
        pgt = cv2.imread(pgt_path, 0)
        imgs.append(ori)
        imgs.append(gt)
        imgs.append(pred)
        imgs.append(pgt)
        img = np.hstack(imgs)
        fig = plt.figure()
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis('off')
        plt.title("Entropy: %s, Prob: %s" % (Entropy[j], probability[j]))
        plt.savefig("/home/hyeonwoo/research/{}/Result/Discriminator/step{}/Result2/".format(str(path), str(
            step)) + os.path.basename(ori_path))
        plt.close()
        j += 1
        if j == len(Entropy):
            break


for i in range(2, 5):
    vis(path, i)

