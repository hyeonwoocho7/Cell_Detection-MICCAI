from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from .utils import CellImageLoad1
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .networks import UNet
import argparse
import sys

sys.path.append('../')
from Discriminator.utils import set_seed, worker_init_fn

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
        default="./Model/Detection/step{}/best.pth".format(str(step)),
    )
    if step > 0:
        parser.add_argument(
            "-lw",
            "--load_weight_path",
            dest="load_weight_path",
            help="load weight path",
            default="./Model/Detection/step{}/best.pth".format(str(step-1)),
        )



    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", default=True, action="store_true"
    )
    if step == 0:
        parser.add_argument(
            "-b", "--batch_size", dest="batch_size", help="batch_size", default=4, type=int
        )
    else:
        parser.add_argument(
            "-b", "--batch_size", dest="batch_size", help="batch_size", default=64, type=int
        )
    parser.add_argument(
        "--visdom", dest="vis", help="visdom show", default=False, type=bool
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=200, type=int
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
        ori_paths = self.gather_path(args.train_path, "ori")
        gt_paths = self.gather_path(args.train_path, "gt")
        data_loader = CellImageLoad1(ori_paths, gt_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn
        )
        self.number_of_traindata = data_loader.__len__()

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
        self.criterion = nn.MSELoss()
        self.losses = []
        self.val_losses = []
        self.evals = []
        self.epoch_loss = 0
        self.bad = 0
        self.vis = args.vis
        self.device = device

    def gather_path(self, train_paths, mode):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
        return ori_paths

    def show_graph(self):
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        plt.savefig("./Model/Detection/step{}/loss.png".format(str(self.step)))
        plt.close()

class TrainNet(_TrainBase):
    def loss_calculate(self, masks_probs_flat, true_masks_flat):
        return self.criterion(masks_probs_flat, true_masks_flat)

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

    def update_vis_plot(self, iteration, loss, window1, update_type):
        self.vis.line(
            X=torch.ones((1)).cpu() * iteration,
            Y=torch.Tensor(loss).unsqueeze(0).cpu(),
            win=window1,
            update=update_type,
        )

    def main(self):
        device = torch.device('cuda:0')
        if self.vis:
            import visdom

            HOSTNAME = "localhost"
            PORT = 8097

            self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env="main")

            vis_title = "loss"
            vis_legend = ["Loss"]
            vis_epoch_legend = ["Loss", "Val Loss"]

            self.iter_plot = self.create_vis_plot(
                "Iteration", "Loss", vis_title, vis_legend
            )
            self.epoch_plot = self.create_vis_plot(
                "Epoch", "Loss", vis_title, vis_epoch_legend
            )
            self.ori_view = self.create_vis_show()
            self.gt_view = self.create_vis_show()
            self.pred_view = self.create_vis_show()

        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))

            pbar = tqdm(total=self.number_of_traindata)
            self.net.train()
            iteration = 1
            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks = data["gt"]

                if self.gpu:
                    imgs = imgs.to(self.device)
                    true_masks = true_masks.to(self.device)

                masks_pred = self.net(imgs)


                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)

                loss = self.loss_calculate(masks_probs_flat, true_masks_flat)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iteration += 1
                if self.vis:
                    self.update_vis_plot(
                        iteration, [loss.item()], self.iter_plot, "append"
                    )

                    self.update_vis_show(imgs.cpu(), self.ori_view)
                    self.update_vis_show(masks_pred, self.pred_view)
                    self.update_vis_show(true_masks.cpu(), self.gt_view)

                pbar.update(self.batch_size)
            pbar.close()
            masks_pred = masks_pred.detach().cpu().numpy()
            cv2.imwrite("conf.tif", (masks_pred * 255).astype(np.uint8)[0, 0])
            loss = self.epoch_loss / (i + 1)
            print("Epoch finished ! Loss: {}".format(loss))
            self.losses.append(loss)
            if epoch % 10 == 0:
                torch.save(
                    self.net.state_dict(),
                    str(
                        self.save_weight_path.parent.joinpath(
                            "epoch_weight/{:05d}.pth".format(epoch)
                        )
                    ),
                )
            if epoch == 100:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
            self.epoch_loss = 0
        self.show_graph()



def Detection(step, seed, path):
    args = parse_args(step)

    args.train_path = [Path(path)]

    args.weight_path = Path(args.weight_path)

    # define model
    net = UNet(n_channels=1, n_classes=1)
    device = torch.device('cuda:0')
    if step > 0:
        net.load_state_dict(torch.load(args.load_weight_path, map_location=device))
    if args.gpu:
        net.to(device)

    args.net = net

    train = TrainNet(args, net, device, step, seed)

    train.main()