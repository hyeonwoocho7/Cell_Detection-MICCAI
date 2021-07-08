from Detection.train import Detection
from Discriminator.train import Discriminator
from Detection.predict import detection_pred
from All_fscore import f_score
from Discriminator.predict import Discriminator_predict
from Discriminator.Dataaugmentation import Dataaugmentation
import os
import shutil
import cv2
import random
from pathlib import Path


# input random seed
seed = 45
random.seed(seed)

# input paths
# test path
test_path = "Data/test_seq2"
# target path
target_path = "Data/seq2"
# source path
base_path = "Data/seq9"
base_ori_paths = base_path / Path("ori")
base_gt_paths = base_path / Path("gt")
source = base_path.split('/')[-1]

steps = 4
for i in range(steps):
    if i == 0:
        Det_Data_path = "Data/Detection/step{}/train".format(str(i))
        os.makedirs("Data/Detection/step{}/train/ori".format(str(i)), exist_ok=True)
        os.makedirs("Data/Detection/step{}/train/gt".format(str(i)), exist_ok=True)

        #random sampling
        locations = []
        y_loc = [int(_) for _ in range(0, 8)]
        for x in range(0, 8):
            y_sampling = random.sample(y_loc, 3)
            for y in y_sampling:
                location = "00{}_00{}".format(str(x), str(y))
                locations.append(location)
        time = ["00000", "00005", "00010", "00015", "00020", "00025", "00030", "00035", "00040", "00045", "00050","00055", "00060", "00065", "00070",
                "00075", "00080", "00085", "00090", "00095"]

        ori_paths = []
        gt_paths = []
        for loc in locations:
            j = random.sample(time, 1)
            ori_paths.append(str(base_ori_paths)+"/{}_{}_{}.tif".format(source, str(j[0]), str(loc)))
            gt_paths.append(str(base_gt_paths)+"/{}_{}_{}.tif".format(source, str(j[0]), str(loc)))


        for t_path in ori_paths:
            ori = cv2.imread(str(base_ori_paths / Path(os.path.basename(t_path))), 0)
            gt = cv2.imread(str(base_gt_paths / Path(os.path.basename(t_path))), 0)
            cv2.imwrite("Data/Detection/step{}/train/ori/".format(str(i))+ os.path.basename(t_path), ori)
            cv2.imwrite("Data/Detection/step{}/train/gt/".format(str(i))+ os.path.basename(t_path), gt)
    Det_Data_path = "Data/Detection/step{}/train".format(str(i))
    Dis_path = "Data/Discriminator/step{}/train".format(str(i))
    os.makedirs(Dis_path, exist_ok=True)
    Dataaugmentation(i, seed, Dis_path, Det_Data_path)
    Detection(i, seed, Det_Data_path)
    Discriminator(i, seed, Dis_path)
    os.makedirs("All_Result/Detection/step{}".format(str(i)), exist_ok=True)
    f_score(i, test_path)
    Det_results = "Result/Detection/step{}".format(str(i))
    os.makedirs(Det_results, exist_ok=True)
    detection_pred(i, target_path)
    shutil.copytree("Data/Detection/step{}/train".format(str(i)),
                    "Data/Detection/step{}/train".format(str(i + 1)))
    os.makedirs("Result/Discriminator/step{}".format(str(i)), exist_ok=True)
    Discriminator_predict(i, seed, Det_results)







