import random
import cv2
import os
import numpy as np
import glob

locations = []
y_loc = [int(_) for _ in range(0, 8)]
for x in range(0, 6):
    y_sampling = random.sample(y_loc, 4)
    for y in y_sampling:
        location = "00{}_00{}".format(str(x), str(y))
        locations.append(location)
time = ["00000", "00005", "00010", "00015", "00020", "00025", "00030", "00035", "00040", "00045", "00050"]

ori_paths = []
gt_paths = []
for loc in locations:
    j = random.sample(time, 1)
    ori_paths.append(
        "/home/hyeonwoo/research/D2_1/ori/D2_1_{}_{}.tif".format(str(j[0]), str(loc)))
for path in ori_paths:
    img = cv2.imread(path, 0)
    cv2.imwrite("/home/hyeonwoo/research/D2_1/random/ori/"+os.path.basename(path), img)

# ps = sorted(glob.glob("/home/hyeonwoo/research/seq1/random/ori/*.tif"))
# for p in ps:
#     img = cv2.imread(p, -1)
#     img = 255*(img/4095)
#     img = img.astype(np.uint8)
#     cv2.imwrite("/home/hyeonwoo/research/seq1/random/ori2/"+os.path.basename(p), img)