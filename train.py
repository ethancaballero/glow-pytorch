from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

import matplotlib.pyplot as plt
import wandb
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch_size_train", default=16, type=int, help="batch size")
parser.add_argument("--batch_size_test", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

parser.add_argument("--iters_per_reconstruct_eval", default=1250, type=int, help="number of samples")
parser.add_argument("--test_set_size", default=10000, type=int, help="number of samples")

parser.add_argument('--wandb_tag', type=str, default="scaling_laws_glow")
parser.add_argument('--wandb_project', type=str, default="scaling_laws_glow")


def sample_data(path, batch_size, image_size, train=True):
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    #"""

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    #import pdb; pdb.set_trace()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


class _Dataset(object):

    def __init__(self, loc, ll=None, transform=None, in_mem=True):
        self.in_mem = in_mem
        self.dataset = torch.load(loc)
        self.lls = None
        self.cond_kl = None
        if ll != None:
            self.lls = torch.load(ll)
        #if in_mem: self.dataset = self.dataset.float().div(255)
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        y = 0
        if self.lls != None:
            y = self.lls[index]
        #if not self.in_mem: x = x.float().div(255)
        #x = self.transform(x) if self.transform is not None else x
        x = x.permute(2,0,1)
        return x, y
#"""

class TeacherFolder(_Dataset):
    TRAIN_LOC = 'data/TeacherFolder/train_32x32.pth'
    TEST_LOC = 'data/TeacherFolder/valid_32x32.pth'

    def __init__(self, root, ll, train=True, transform=None):
        return super(TeacherFolder, self).__init__(root, ll, transform)

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer):
    dataset = iter(sample_data(args.path, args.batch_size_train, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
        ]
    )

    #"""
    train_data = TeacherFolder('/home/mila/c/caballero/research/scaling_outer/scaling_laws/minGPT/teacher_data/train_teacher_imgs.pth', ll='/home/mila/c/caballero/research/scaling_outer/scaling_laws/minGPT/teacher_data/train_teacher_lls.pth', train=True, transform=transform)
    test_data  = TeacherFolder('/home/mila/c/caballero/research/scaling_outer/scaling_laws/minGPT/teacher_data/val_teacher_imgs.pth', ll='/home/mila/c/caballero/research/scaling_outer/scaling_laws/minGPT/teacher_data/val_teacher_lls.pth', train=False, transform=transform)
    #"""

    """
    train_data = TeacherFolder('/Users/ethancaballero/research/scaling_breadth/create_dataset/data_1M/train_teacher_imgs.pth', ll='/Users/ethancaballero/research/scaling_breadth/create_dataset/data_1M/train_teacher_lls.pth', train=True, transform=transform)
    test_data = TeacherFolder('/Users/ethancaballero/research/scaling_breadth/create_dataset/data_1M/val_teacher_imgs.pth', ll='/Users/ethancaballero/research/scaling_breadth/create_dataset/data_1M/val_teacher_lls.pth', train=False, transform=transform)
    #"""

    test_data.dataset = test_data.dataset[:args.test_set_size]
    test_data.lls = test_data.lls[:args.test_set_size]

    train_loader = DataLoader(
                train_data, shuffle=True, batch_size=args.batch_size_train, num_workers=4
            )
    test_loader = DataLoader(
                test_data, shuffle=True, batch_size=args.batch_size_test, num_workers=4
            )

    wandb.init(project=args.wandb_project, reinit=True, tags=[args.wandb_tag])
    wandb.config.update(args)
    wandb.config.update({"params": sum(p.numel() for p in model.parameters())})

    #with tqdm(range(args.iter)) as pbar:
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (image, ll) in pbar:
        #image, _ = next(dataset)
        image = image.to(device)

        image = image * 255

        if args.n_bits < 8:
            image = torch.floor(image / 2 ** (8 - args.n_bits))

        image = image / n_bins - 0.5

        if i == 0:
            with torch.no_grad():
                log_p, logdet, _ = model.module(
                    image + torch.rand_like(image) / n_bins
                )

                continue

        else:
            log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

        logdet = logdet.mean()

        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
        model.zero_grad()
        loss.backward()
        # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr
        optimizer.step()

        pbar.set_description(
            f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
        )

        if i % args.iters_per_reconstruct_eval == 0:
            test_maes = []
            test_mses = []
            test_loss = []
            test_lls = []
            for _it, (image, ll) in enumerate(test_loader):
                test_lls.append(ll.float())
                image = image.to(device)

                image = image * 255

                if args.n_bits < 8:
                    image = torch.floor(image / 2 ** (8 - args.n_bits))

                image = image / n_bins - 0.5
                with torch.no_grad():
                    log_p, logdet, z = model(image + torch.rand_like(image) / n_bins)
                    loss_test, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
                    image_rec = model_single.reverse(z).cpu().data
                    #import pdb; pdb.set_trace()
                    image_diff = image.cuda() - image_rec.cuda()
                    rec_mae = (image_diff).abs()
                    rec_mae = torch.flatten(rec_mae, 1, 3).mean(-1)
                    rec_mse = (image_diff) ** 2
                    rec_mse = torch.flatten(rec_mse, 1, 3).mean(-1)
                    test_maes.append(rec_mae)
                    test_mses.append(rec_mse)

            ll_cat = torch.cat(test_lls).float().cpu()
            rec_mae_cat = torch.cat(test_maes).float().cpu()
            rec_mse_cat = torch.cat(test_mses).float().cpu()
            sort_idxs = ll_cat.float().sort(descending=True)[1]
            ll_cat, rec_mae_cat, rec_mse_cat = ll_cat[sort_idxs], rec_mae_cat[sort_idxs], rec_mse_cat[sort_idxs]
            figure, axis = plt.subplots(1, 2)
            markersize = 2
            axis[0].plot(np.arange(0, len(rec_mae_cat)), rec_mae_cat.cpu().numpy(), '.', markersize=markersize, color='k')
            axis[1].plot(np.arange(0, len(rec_mse_cat)), rec_mse_cat.cpu().numpy(), '.', markersize=markersize, color='k')
            axis[0].set_ylabel("reconstuction mae")
            axis[1].set_ylabel("reconstuction mse")
            
            log_dict = {}
            log_dict.update({'eval_loss': loss_test})
            log_dict.update({"charts__ll/"+str(i*args.batch_size_train)+"_training_samples": figure})
            wandb.log(log_dict, step=i*args.batch_size_train)



        """
        if i % 100 == 0:
            with torch.no_grad():
                utils.save_image(
                    model_single.reverse(z_sample).cpu().data,
                    f"sample/{str(i + 1).zfill(6)}.png",
                    normalize=True,
                    nrow=10,
                    range=(-0.5, 0.5),
                )

        if i % 10000 == 0:
            torch.save(
                model.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
            )
            torch.save(
                optimizer.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
            )
            #"""

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    print("params: ", sum(p.numel() for p in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)
