from datetime import datetime
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import cv2
from .networks import UNet
from .utils import local_maxima, make_pgt, optimum, target_peaks_gen, remove_outside_plot
import argparse
import os
import random


def parse_args(step, test_path):
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="dataset's path",
        default=test_path,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./Result/Detection/step{}".format(str(step)),
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        default="./Model/Detection/step{}/best.pth".format(str(step)),
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", default=True, action="store_true"
    )

    args = parser.parse_args()
    return args


class Predict:
    def __init__(self, args):
        self.net = args.net
        self.gpu = args.gpu

        self.ori_path = args.input_path / Path("ori")

        self.save_ori_path = args.output_path / Path("ori")
        self.save_pred_path = args.output_path / Path("pred")
        self.save_pgt_path = args.output_path / Path("pgt")


        self.save_ori_path.mkdir(parents=True, exist_ok=True)
        self.save_pred_path.mkdir(parents=True, exist_ok=True)
        self.save_pgt_path.mkdir(parents=True, exist_ok=True)

    def pred(self, ori):
        img = (ori.astype(np.float32) / 255).reshape(
            (1, ori.shape[0], ori.shape[1])
            )

        with torch.no_grad():
            #numpy to tensor
            img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                img = img.cuda()
            mask_pred = self.net(img)
        #tesor to numpy
        pre_img = mask_pred.detach().cpu().numpy()[0, 0]
        pre_img = (pre_img * 255).astype(np.uint8)
        return pre_img

    def main(self):
        self.net.eval()
        # path def
        paths = sorted(self.ori_path.glob("*.tif"))
        for i, path in enumerate(paths):
            ori = cv2.imread(str(path), 0)
            pre_img = self.pred(ori)
            pgt_img = make_pgt(pre_img, threshold=100, dist=4)

            cv2.imwrite(str(self.save_pred_path / Path(os.path.basename(str(path)))), pre_img)
            cv2.imwrite(str(self.save_ori_path / Path(os.path.basename(str(path)))), ori)
            cv2.imwrite(str(self.save_pgt_path / Path(os.path.basename(str(path)))), pgt_img)






def detection_pred(step, test_path):
    args = parse_args(step, test_path)

    args.input_path = Path(args.input_path)
    args.output_path = Path(args.output_path)

    net = UNet(n_channels=1, n_classes=1)
    device = torch.device('cuda:0')
    net.load_state_dict(torch.load(args.weight_path, map_location=device))

    if args.gpu:
        net.cuda()
    args.net = net

    pred = Predict(args)

    pred.main()
