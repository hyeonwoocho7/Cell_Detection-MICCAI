from datetime import datetime
import torch
import numpy as np
from pathlib import Path
import cv2
from .resnet_dropout import *
import argparse
import os
from .utils import set_seed, worker_init_fn
from collections import OrderedDict
import glob
import random
import os
import random



def parse_args(step):
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    if step > 0:
        parser.add_argument(
            "-i",
            "--input_path",
            dest="input_path",
            help="dataset's path",
            default="./Result/Discriminator/step{}/remove_pgt".format(str(step-1)),
            type=str,
        )
    else:
        parser.add_argument(
            "-i",
            "--input_path",
            dest="input_path",
            help="dataset's path",
            default="./Result/Detection/step{}/".format(str(step)),
            type=str,
        )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./Result/Discriminator/step{}".format(str(step)),
        type=str,
    )

    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        default="./Model/Discriminator/step{}/best.pth".format(str(step)),
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true"
    )

    args = parser.parse_args()
    return args


class Predict:
    def __init__(self, args, net, Detection_path, step, seed):
        set_seed(seed)

        self.net = net
        self.gpu = args.gpu
        self.step = step
        self.ori_path = args.input_path / Path("ori")
        self.pgt_path = args.input_path / Path("pgt")


        self.save_sigmoid = args.output_path / Path("sigmoid.txt")
        self.save_entropy = args.output_path / Path("Entropy.txt")
        self.save_sorted_entropy = args.output_path / Path("Sorted_Entropy.txt")
        self.save_selected_pgt = args.output_path / Path("save_selected_top")
        self.select_ori = self.save_selected_pgt / Path("ori")
        self.select_pgt = self.save_selected_pgt / Path("pgt")
        self.remove = args.output_path / Path("remove_pgt")
        self.remove_ori = self.remove / Path("ori")
        self.remove_pgt = self.remove / Path("pgt")
        self.Detection_ori = Detection_path / Path("ori")
        self.Detection_pgt = Detection_path / Path("pgt")

        self.save_selected_pgt.mkdir(parents=True, exist_ok=True)
        self.select_ori.mkdir(parents=True, exist_ok=True)
        self.select_pgt.mkdir(parents=True, exist_ok=True)
        self.remove.mkdir(parents=True, exist_ok=True)
        self.remove_ori.mkdir(parents=True, exist_ok=True)
        self.remove_pgt.mkdir(parents=True, exist_ok=True)



    def pred(self, ori, pred, ori_path):
        img = (ori.astype(np.float32) / 255).reshape(
            (1, ori.shape[0], ori.shape[1])
            )
        pred = (pred.astype(np.float32) / 255).reshape(
            (1, pred.shape[0], pred.shape[1])
            )

        with torch.no_grad():
            #numpy to tensor
            img = torch.from_numpy(img).unsqueeze(0)
            pred = torch.from_numpy(pred).unsqueeze(0)
            if self.gpu:
                img = img.cuda()
                pred = pred.cuda()
            y1 = []
            for i in range(10):
                inputs = torch.cat([img, pred], dim=1)
                output1 = self.net(inputs)
                output2 = torch.sigmoid(output1)
                y1.append(output2)
            confidence = sum(y1) / len(y1)
            predict = confidence > 0.5
            confidence = confidence.data.numpy()
            Entropy = -confidence*np.log2(confidence)
            with self.save_entropy.open(mode="a") as f:
                f.write("%.12f, %.12f, %s\n" % (Entropy, confidence, ori_path))







    def enable_dropout(self, m):
        for m in self.net.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def main(self):
        self.net.eval()
        self.enable_dropout(self.net)
        # path def
        ori_paths = sorted(self.ori_path.glob("*.tif"))
        pgt_paths = sorted(self.pgt_path.glob("*.tif"))
        for i, (ori_path, pred_path) in enumerate(zip(ori_paths, pgt_paths)):
            ori = cv2.imread(str(self.Detection_ori) + '/' + os.path.basename(ori_path), 0)
            pred = cv2.imread(str(self.Detection_pgt) + '/' + os.path.basename(pred_path), 0)
            self.pred(ori, pred, ori_path)
        f = open(self.save_entropy, 'r')
        data = f.read()
        datas = data.split("\n")
        datas = sorted(datas[:-1])
        entropy = []
        sigmoid = []
        paths = []
        positive = []
        for i, data in enumerate(datas):
            data = data.split(', ')
            if float(data[1]) > 0.5:
                positive.append(datas[i])
        for j, p in enumerate(positive):
            p = p.split(', ')
            entropy.append(float(p[0]))
            sigmoid.append(float(p[1]))
            paths.append(p[2])

        with open(self.save_sorted_entropy, mode='wt',encoding='utf-8') as myfile:
            myfile.write('\n'.join(positive))

        for k, path in enumerate(paths[:int(0.1*len(paths))]):
            ori = cv2.imread(str(self.Detection_ori / Path(os.path.basename(path))), 0)
            pgt = cv2.imread(str(self.Detection_pgt / Path(os.path.basename(path))), 0)

            cv2.imwrite(str(self.select_ori / Path(os.path.basename(path))), ori)
            cv2.imwrite(str(self.select_pgt / Path(os.path.basename(path))), pgt)

        ori_ps = sorted(self.select_ori.glob("*.tif"))

        for p in ori_ps:
            ori = cv2.imread(str(p), 0)
            gt = cv2.imread(str(self.select_pgt / Path(os.path.basename(p))), 0)
            cv2.imwrite("./Data/Detection/step{}/train/ori/".format(str(self.step + 1)) + os.path.basename(p), ori)
            cv2.imwrite("./Data/Detection/step{}/train/gt/".format(str(self.step + 1)) + os.path.basename(p), gt)
        # remove prior-step pseudo-label
        before = sorted(self.ori_path.glob("*.tif"))
        top = sorted(self.select_ori.glob("*.tif"))
        b_p = []
        t_p = []
        for i in before:
            b_p.append(os.path.basename(i))
        for j in top:
            t_p.append(os.path.basename(j))

        new_paths = list(set(b_p) - set(t_p))
        for path in new_paths:
            ori = cv2.imread(str(self.Detection_ori / Path(os.path.basename(path))), 0)
            pgt = cv2.imread(str(self.Detection_pgt / Path(os.path.basename(path))), 0)
            cv2.imwrite(str(self.remove_ori / Path(os.path.basename(path))), ori)
            cv2.imwrite(str(self.remove_pgt / Path(os.path.basename(path))), pgt)


def Discriminator_predict(step, seed, path):

    args = parse_args(step)
    args.input_path = Path(args.input_path)
    args.output_path = Path(args.output_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def fix_model_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict


    # load it
    state_dict = torch.load(args.weight_path, map_location=device)
    net = resnet18(pretrained=False)
    net.load_state_dict(fix_model_state_dict(state_dict))



    if args.gpu:
        net.cuda()
    args.net = net

    pred = Predict(args, net, path, step, seed)
    pred.main()


