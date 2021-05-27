from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from pathlib import Path
from .resnet_dropout import *
import matplotlib.pyplot as plt
from .utils import set_seed, worker_init_fn
import numpy as np
import argparse
from .load import CellImage
from collections import OrderedDict
import os
import random


def parse_args(step):
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-t",
        "--train_path",
        dest="train_path",
        help="training dataset's path",
        default="-1",
        type=str,
    )

    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="save weight path",
        default="./Model/Discriminator/step{}/best.pth".format(str(step)),
    )
    if step > 0:
        parser.add_argument(
            "-lw",
            "--load_weight_path",
            dest="load_weight_path",
            help="load weight path",
            default="./Model/Discriminator/step{}/best.pth".format(str(step-1)),
        )

    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", default=True, action="store_true"
    )
    if step == 0:
        parser.add_argument(
            "-b", "--batch_size", dest="batch_size", help="batch_size", default=64, type=int
        )
    else:
        parser.add_argument(
            "-b", "--batch_size", dest="batch_size", help="batch_size", default=1024, type=int
        )
    parser.add_argument(
        "--visdom", dest="vis", help="visdom show", default=False, type=bool
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=51, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )

    args = parser.parse_args()
    return args


class _TrainBase:
    def __init__(self, args, net, device, step, seed):
        set_seed(seed)
        train_dir_paths = args.train_path
        train_data_loader = CellImage(train_dir_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            train_data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn
        )
        self.number_of_traindata = train_data_loader.__len__()


        self.save_weight_path = args.weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_weight_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
                args.epochs, args.batch_size, args.learning_rate, args.gpu
            )
        )

        self.net = net
        self.step = step
        self.train = None
        self.val = None

        self.N_train = None
        self.optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.criterion = nn.BCEWithLogitsLoss()
        self.losses = []
        self.acces = []
        self.val_losses = []
        self.val_acces = []
        self.evals = []
        self.epoch_loss = 0
        self.epoch_acc = 0
        self.bad = 0
        self.vis = args.vis
        self.device = device


    def gather_path(self, train_paths, mode):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
        return ori_paths

    def show_loss_graph(self):
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        plt.savefig("./Model/Discriminator/step{}/loss.png".format(self.step))
        plt.close()
    def show_acc_graph(self):
        x = list(range(len(self.acces)))
        plt.plot(x, self.acces)
        #plt.plot(x, self.val_acces)
        #plt.show()
        plt.savefig("./Model/Discriminator/step{}/acc.png".format(self.step))
        plt.close()



class TrainNet(_TrainBase):
    def create_vis_show(self):
        return self.vis.images(
            torch.ones((self.batch_size, 1, 256, 256)), self.batch_size
        )

    def update_vis_show(self, images, window1):
        self.vis.images(images, self.batch_size, win=window1)

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend),
        )

    def update_vis_loss_plot(self, iteration, loss, window1, update_type):
        self.vis.line(
            X=torch.ones((1)).cpu() * iteration,
            Y=torch.Tensor(loss).unsqueeze(0).cpu(),
            win=window1,
            update=update_type,
        )
    def update_vis_acc_plot(self, iteration, acc, window1, update_type):
        self.vis.line(
            X=torch.ones((1)).cpu() * iteration,
            Y=torch.Tensor(acc).unsqueeze(0).cpu(),
            win=window1,
            update=update_type,
        )


    def main(self):
        if self.vis:
            import visdom

            HOSTNAME = "localhost"
            PORT = 8097

            self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env="main3")

            vis_title = "loss"
            vis_loss_legend = ["Loss"]
            vis_acc_legend = ["acc"]
            vis_epoch_loss_legend = ["Loss", "Val Loss"]
            vis_epoch_acc_legend = ["Acc", "Val acc"]

            self.iter_loss_plot = self.create_vis_plot(
                "Iteration", "Loss", vis_title, vis_loss_legend
            )
            self.iter_acc_plot = self.create_vis_plot(
                "Iteration", "acc", vis_title, vis_acc_legend
            )
            self.epoch_loss_plot = self.create_vis_plot(
                "Epoch", "Loss", vis_title, vis_epoch_loss_legend
            )
            self.epoch_acc_plot = self.create_vis_plot(
                "Epoch", "acc", vis_title, vis_epoch_acc_legend
            )
            self.ori_view = self.create_vis_show()
            self.gt_view = self.create_vis_show()


        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))

            pbar = tqdm(total=self.number_of_traindata)

            self.net.train()
            iteration = 1
            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                masks = data["gt"]
                labels = data["label"]


                self.optimizer.zero_grad()

                if self.gpu:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.float().to(self.device)

                inputs = torch.cat([imgs, masks], dim=1)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, labels)
                acc = ((torch.sigmoid(outputs) > 0.5) == labels.byte()).float().mean()

                self.epoch_loss += loss.item()
                self.epoch_acc += acc.item()

                loss.backward()
                self.optimizer.step()

                iteration += 1
                if self.vis:
                    self.update_vis_loss_plot(
                        iteration, [loss.item()], self.iter_loss_plot, "append"
                    )
                    self.update_vis_acc_plot(
                        iteration, [acc.item()], self.iter_acc_plot, "append"
                    )

                    self.update_vis_show(imgs.cpu(), self.ori_view)
                    self.update_vis_show(masks.cpu(), self.gt_view)


                pbar.update(self.batch_size)
            pbar.close()
            loss = self.epoch_loss / (i + 1)
            acc = self.epoch_acc / (i + 1)

            print("Epoch finished ! Loss: {} acc: {}".format(loss, acc))

            self.losses.append(loss)
            self.acces.append(acc)

            if epoch % 10 == 0:
                torch.save(
                    self.net.state_dict(),
                    str(
                        self.save_weight_path.parent.joinpath(
                            "epoch_weight/{:05d}.pth".format(epoch)
                        )
                    ),
                )
            if epoch == 50:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
            self.epoch_loss = 0
            self.epoch_acc = 0
        self.show_loss_graph()
        self.show_acc_graph()



def Discriminator(step, seed, train_path):
    args = parse_args(step)

    args.train_path = [train_path]
    args.weight_path = Path(args.weight_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fix_model_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict

    if step > 0:
        # load it
        state_dict = torch.load(args.load_weight_path, map_location=device)
        # define model
        net = resnet18(pretrained=False)
        net.load_state_dict(fix_model_state_dict(state_dict))
    else:
        net = resnet18(pretrained=False)
    if args.gpu:
        net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    args.net = net


    train = TrainNet(args, net, device, step, seed)

    train.main()