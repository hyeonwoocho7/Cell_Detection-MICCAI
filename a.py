import glob
import os
import random
from pathlib import Path

y_loc = [int(_) for _ in range(0, 8)]
print(y_loc)

base_path = "/home/hyeonwoo/research/Experiment/no_control"
base_ori_paths = base_path / Path("ori")
n = 20
ori_paths = sorted(base_ori_paths.glob("*.tif"))
sampling = random.sample(ori_paths, n)

locations = []
for s in sampling:
    basename = os.path.basename(s).split(".")[0]
    parts = basename.split("_")
    #time = parts[1]
    location = f"{parts[2]}_{parts[3]}"
    locations.append(location)


time = ["00000", "000005", "00010", "00015", "00020", "00025", "00030", "00035", "00040", "00045", "00050", "00055", "00060", "00065", "00070",
        "00075", "00080", "00085", "00090", "00095"]

ori_paths = []
gt_paths = []
for i in locations:
    j = random.sample(time, 1)
    time = set(time) - set(j)
    ori_paths.append("/home/hyeonwoo/research/Experiment/no_control/ori/seq2_{}_{}.tif".format(str(j[0]), str(i)))
    gt_paths.append("/home/hyeonwoo/research/Experiment/no_control/ori/seq2_{}_{}.tif".format(str(j[0]), str(i)))

print(gt_paths)

