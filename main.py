import argparse
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


def get_args_parser():
    parser = argparse.ArgumentParser("Detection", add_help=False)
    parser.add_argument(
        "--random_seed", default=45, type=int, help="Random Seed"
    )
    parser.add_argument(
        "--test_path", default="Data/test_seq2", type=str, help="test dataset path"
    )
    parser.add_argument(
        "--target_path", default="Data/seq2", type=str, help="target domain dataset path"
    )
    parser.add_argument(
        "--source_path", default="Data/seq9", type=str, help="source domain dataset path"
    )
    parser.add_argument(
        "--steps", default=4, type=int, help="steps for domain extension"
    )
    parser.add_argument(
        "--det_train_folder", default="Data/Detection/step{}/train/", type=str, help="Detection folder path"
    )
    parser.add_argument(
        "--dis_train_folder", default="Data/Discriminator/step{}/train/", type=str, help="Discriminator folder path"
    )
    parser.add_argument(
        "--det_result_folder", default="Result/Detection/step{}", type=str, help="Detection result path"
    )
    parser.add_argument(
        "--dis_result_folder", default="Result/Discriminator/step{}", type=str, help="Discriminator result path"
    )
    parser.add_argument(
        "--All_result_path", default="All_Result/Detection/step{}", type=str, help="All result path"
    )
    
    
    return parser


    
def main(args):
    seed = args.random_seed
    random.seed(seed)
    
    # test path
    test_path = args.test_path
    # target path
    target_path = args.target_path
    # source path
    source_path = args.source_path
    source_ori_paths = source_path / Path("ori")
    source_gt_paths = source_path / Path("gt")
    source = source_path.split('/')[-1]
    
    for i in range(args.steps):
        Det_Data_path = args.det_train_folder.format(str(i))
        Dis_Data_path = args.dis_train_folder.format(str(i))
        Det_origin_path = str(Det_Data_path)+"ori"
        Det_gt_path = str(Det_Data_path)+"gt" 
        os.makedirs(Det_origin_path.format(str(i)), exist_ok=True)
        os.makedirs(Det_gt_path.format(str(i)), exist_ok=True)
        
        if i == 0:
            #random sampling
            locations = []
            y_loc = [int(_) for _ in range(0, 8)]
            for x in range(0, 8):
                y_sampling = random.sample(y_loc, 3)
                for y in y_sampling:
                    location = "00{}_00{}".format(str(x), str(y))
                    locations.append(location)
            time = ["00000", "00005", "00010", "00015", "00020", 
                    "00025", "00030", "00035", "00040", "00045",
                    "00050", "00055", "00060", "00065", "00070",
                    "00075", "00080", "00085", "00090", "00095"]

            ori_paths = []
            gt_paths = []
            for loc in locations:
                j = random.sample(time, 1)
                ori_paths.append(str(source_ori_paths)+"/{}_{}_{}.tif".format(source, str(j[0]), str(loc)))
                gt_paths.append(str(source_gt_paths)+"/{}_{}_{}.tif".format(source, str(j[0]), str(loc)))


            for t_path in ori_paths:
                ori = cv2.imread(str(source_ori_paths / Path(os.path.basename(t_path))), 0)
                gt = cv2.imread(str(source_gt_paths / Path(os.path.basename(t_path))), 0)
                cv2.imwrite(Det_origin_path+ os.path.basename(t_path), ori)
                cv2.imwrite(Det_gt_path.format(str(i))+ os.path.basename(t_path), gt)
    
        Dataaugmentation(i, seed, Dis_Data_path, Det_Data_path)
        Detection(i, seed, Det_Data_path)
        Discriminator(i, seed, Dis_Data_path)
        os.makedirs(args.All_result_path.format(str(i)), exist_ok=True)
        f_score(i, test_path)
        Det_results = args.det_result_folder.format(str(i))
        os.makedirs(Det_results, exist_ok=True)
        detection_pred(i, target_path)
        shutil.copytree(args.det_train_folder.format(str(i)),
                        args.det_train_folder.format(str(i + 1)))
        os.makedirs(args.dis_result_folder.format(str(i)), exist_ok=True)
        Discriminator_predict(i, seed, Det_results)
        
    

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    
    main(args)
    


