# System / Python
# python main.py -m 'train' -trsr 0.03 -vsr 0.01
# python main.py -m 'test' -tesr 0.01

import os
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from net import MRIReconNet as MRInet

from IXI_dataset import IXIData as Dataset
from mri_tools import rA, rAtA, rfft2_1, rifft2_1, rfft2
from utils import psnr_slice, ssim_slice, mse_slice

import matplotlib.pyplot as plt
import cv2
import matplotlib
import seaborn as sns

from skimage import transform
from torchprofile import profile

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default='tcp://localhost:1836', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(), help='number of gpus per node')
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=1, help='number of iterations') # 9
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=2, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=106, help='maximum number of epochs')
# parameters related to data and masks
'''parser.add_argument('--train-path', type=str, default='data/IXI_T2_FULL/train', help='path of training data')
parser.add_argument('--val-path', type=str, default='data/IXI_T2_FULL/val', help='path of validation data')
parser.add_argument('--test-path', type=str, default='data/IXI_T2_FULL/test', help='path of test data')     # default='data/IXI_T2_FULL/test'
parser.add_argument('--train1-path', type=str, default='data/PD_FULL/train', help='path of training data else')
parser.add_argument('--val1-path', type=str, default='data/PD_FULL/val', help='path of validation data else')
parser.add_argument('--test1-path', type=str, default='data/PD_FULL/test', help='path of test data else')  ''' # default='data/PD_FULL/test'

parser.add_argument('--train-path', type=str, default='data/PD_FULL/train', help='path of training data')
parser.add_argument('--val-path', type=str, default='data/PD_FULL/val', help='path of validation data')
parser.add_argument('--test-path', type=str, default='data/PD_FULL/test', help='path of test data')
parser.add_argument('--train1-path', type=str, default='data/IXI_T2_FULL/train', help='path of training data else')
parser.add_argument('--val1-path', type=str, default='data/IXI_T2_FULL/val', help='path of validation data else')
parser.add_argument('--test1-path', type=str, default='data/IXI_T2_FULL/test', help='path of test data else')

parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask - random0.2.mat', help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.001, help='sampling rate of training data')
parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.02, help='sampling rate of validation data')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.02, help='sampling rate of test data')
# save path
parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')
parser.add_argument('--loss-curve-path', type=str, default='./runs/loss_curve/', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')


def to_complex(data):   # k-space数据
    # from data[1,256,256,2] to [1,256,256]complex
    data = data[:, :, :, 0] + 1j * data[:, :, :, 1]
    # data = cv2.magnitude(data[:, :, :, 0], data[:, :, :, 1])
    return data


def sobel_plus(image):  # sobel_plus边缘增强
    a = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    sobel_x = a * torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = a * torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    sobel_diagonal1 = a * torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)
    sobel_diagonal2 = a * torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    sobel_diagonal1 = sobel_diagonal1.view(1, 1, 3, 3)
    sobel_diagonal2 = sobel_diagonal2.view(1, 1, 3, 3)

    sobel_x = torch.cat((sobel_x, sobel_x), axis=1)
    sobel_y = torch.cat((sobel_y, sobel_y), axis=1)
    sobel_diagonal1 = torch.cat((sobel_diagonal1, sobel_diagonal1), axis=1)
    sobel_diagonal2 = torch.cat((sobel_diagonal2, sobel_diagonal2), axis=1)

    grad_x = torch.nn.functional.conv2d(image, sobel_x.cuda(), stride=(1, 1), padding=(1, 1))
    grad_y = torch.nn.functional.conv2d(image, sobel_y.cuda(), stride=(1, 1), padding=(1, 1))
    grad_diagonal1 = torch.nn.functional.conv2d(image, sobel_diagonal1.cuda(), stride=(1, 1), padding=(1, 1))
    grad_diagonal2 = torch.nn.functional.conv2d(image, sobel_diagonal2.cuda(), stride=(1, 1), padding=(1, 1))

    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_diagonal1 ** 2 + grad_diagonal2 ** 2)
    return edge


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args):
    assert mode in ['train', 'val', 'test']
    loss, psnr_up, ssim_up, mse_up, psnr_down, ssim_down, mse_down = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        label = data_batch[0].to(rank, non_blocking=True)
        mask_under = data_batch[1].to(rank, non_blocking=True)
        mask_net_up = data_batch[2].to(rank, non_blocking=True)
        mask_net_down = data_batch[3].to(rank, non_blocking=True)
        fname = data_batch[4]   # 重建图像文件名
        slice_id = data_batch[5]    # 重建切片索引
        # 参考模态对应切片
        refer = data_batch[6].to(rank, non_blocking=True)

        ##### PJS数据集 #####
        # torch.from_numpy(transform.resize(label.cpu(), (2, 256, 256, 1))).cuda()
        # refer = torch.from_numpy(transform.resize(refer.cpu(), (2, 256, 256, 1))).cuda()

        '''
        plt.figure(1)
        plt.subplot(1, 2, 1), plt.imshow(label.cpu().squeeze(), cmap='gray'), plt.title('IXI_T2')
        plt.subplot(1, 2, 2), plt.imshow(refer.cpu().squeeze(), cmap='gray'), plt.title('PD')
        plt.show()
        '''

        '''plt.figure(1)
        plt.imshow(np.rot90(label.cpu().squeeze()), cmap='gray'), plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.savefig("figs/06_label.png", bbox_inches='tight', pad_inches=0)  # plt.title('label')
        # plt.show()
        plt.figure(2)
        plt.imshow(np.rot90(refer.cpu().squeeze()), cmap='gray'), plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.savefig("result_image/refer.eps", bbox_inches='tight', pad_inches=0, dpi=1200)  # plt.title('refer')
        plt.show()'''

        under_img = rAtA(label, mask_under)  # 初始欠采样imgae
        under_kspace = rA(label, mask_under)  # 初始欠采样kspace

        '''plt.figure(3)  # 初始输入欠采样image
        plt.imshow(np.rot90(under_img.cpu().squeeze()), cmap='gray')
        # plt.title("under_sampled image")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("save/under_sampled image.png", bbox_inches='tight', pad_inches=0)
        plt.figure(4)  # 初始输入欠采样kspace
        plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(under_kspace.cpu().detach().numpy()).squeeze()))), cmap='gray')
        # plt.title("under_sampled kspace")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("save/under_sampled kspace.png", bbox_inches='tight', pad_inches=0)
        plt.figure(5)  # 初始输入配对参考模态image
        plt.imshow(np.rot90(refer.cpu().squeeze()), cmap='gray')
        # plt.title("refer image")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("save/refer image.png", bbox_inches='tight', pad_inches=0)
        plt.show()'''

        # up分支网络 - kspace[1, 256, 256, 2]
        net_kspace_up = rA(label, mask_net_up)
        # down分支网络 - image[1, 256, 256, 1]
        net_img_down = rAtA(label, mask_net_down)
        # 加上虚部image[1, 256, 256, 2]
        net_img_down = torch.cat([net_img_down, torch.zeros_like(net_img_down)], dim=-1)
        # refer参考模态加上虚部[1, 256, 256, 2]
        refer_img = torch.cat([refer, torch.zeros_like(refer)], dim=-1)
        # label全采样ground truth加上虚部[1, 256, 256, 2]
        label_extend = torch.cat([label, torch.zeros_like(label)], dim=-1)

        '''plt.figure(6)  # 经过随机掩膜的kspace
        plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(net_kspace_up.cpu().detach().numpy()).squeeze()))), cmap='gray')
        # plt.title("select kspace")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("save/select kspace.png", bbox_inches='tight', pad_inches=0)
        plt.figure(7)  # 经过随机掩膜的image
        plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(net_img_down.cpu().detach().numpy()).squeeze()))), cmap='gray')
        # plt.title("select image")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("save/select image.png", bbox_inches='tight', pad_inches=0)

        plt.figure(8)  # kspace的随机掩膜
        plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(mask_net_up.cpu().detach().numpy()).squeeze()))), cmap='gray')
        # plt.title("kspace mask")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("save/kspace mask.png", bbox_inches='tight', pad_inches=0)
        plt.figure(9)  # image的随机掩膜
        plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(mask_net_down.cpu().detach().numpy()).squeeze()))), cmap='gray')
        # plt.title("image mask")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("save/image mask.png", bbox_inches='tight', pad_inches=0)
        plt.show()'''

        if mode == 'test':
            net_kspace_up = under_kspace  # 上分支
            net_img_down = under_img  # 下分支
            net_img_down = torch.cat([net_img_down, torch.zeros_like(net_img_down)], dim=-1)
            mask_net_up = mask_net_down = mask_under

        output_up, loss_layers_up, output_down, loss_layers_down = model(net_kspace_up.permute(0, 3, 1, 2).contiguous(),
                                                                         mask_net_up,
                                                                         net_img_down.permute(0, 3, 1, 2).contiguous(),
                                                                         mask_net_down,
                                                                         refer_img.permute(0, 3, 1, 2).contiguous())
        output_up, output_down = output_up.permute(0, 2, 3, 1).contiguous(), output_down.permute(0, 2, 3, 1).contiguous()

        # kspace输出 [B, 256, 256, 2]
        output_up_kspace = output_up
        output_down_kspace = rfft2_1(output_down)

        # image输出 [B, 256, 256, 2]
        output_up_image = rifft2_1(output_up)
        output_down_image = output_down

        output_up_real = torch.view_as_complex(output_up_image).float().unsqueeze(-1)
        output_down_real = torch.view_as_complex(output_down).float().unsqueeze(-1)
        ot_img = to_complex(output_down.cpu().detach().numpy())
        ot = np.log(1 + np.abs(ot_img[0, :, :]))  # ot[256, 256]

        '''plt.figure(2)  # 初始输入欠采样image
        plt.imshow(np.rot90(under_img.cpu().squeeze()), cmap='gray')
        # plt.title("under_sampled image")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.savefig("T2w_image/zero_padding.png", bbox_inches='tight', pad_inches=0)'''

        '''plt.figure(3)  # 重建图像
        plt.imshow(np.rot90(((output_up_real+output_down_real)/2).cpu().squeeze()), cmap='gray'), plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("T2w_image/8(1).png", bbox_inches='tight', pad_inches=0)  # ,plt.title('reconstruction')

        plt.figure(4)  # error map
        error = np.rot90(label.cpu() - ((output_up_real+output_down_real)/2).cpu()).squeeze()
        norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1)
        plt.imshow(error, cmap='bwr', norm=norm)
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.savefig("T2w_image/8(1)_error.png", bbox_inches='tight', pad_inches=0)  # ,plt.title('error')
        plt.colorbar()
        plt.show()'''


        if mode == 'test':
            # print('file_name:{}'.format(fname))
            # print('slice_id:{}'.format(slice_id))

            '''plt.figure(11)  # 重建结果kspace
            plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(output_up.cpu().detach().numpy()).squeeze()))),
                       cmap='gray')
            # plt.title("kspace reconstruction")
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            # plt.savefig("save/kspace reconstruction.png", bbox_inches='tight', pad_inches=0)
            plt.figure(12)  # 重建结果image
            plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(output_down.cpu().detach().numpy()).squeeze()))),
                       cmap='gray')
            # plt.title("image reconstruction")
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.savefig("result_image/image reconstruction.png", bbox_inches='tight', pad_inches=0)
            plt.figure(13)  # 输入groundtruth image
            plt.imshow(np.rot90(label.cpu().squeeze()), cmap='gray')
            # plt.title("groundtruth")
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.savefig("result_image/image groundtruth.png", bbox_inches='tight', pad_inches=0)
            plt.show()'''

            '''plt.figure(1)  # 输入欠采样 输出重建 ground_truth
            # batch size = 1
            plt.subplot(1, 4, 1), plt.imshow(np.rot90(net_img_down_0.cpu().squeeze()), cmap='gray'), plt.title('input')
            plt.subplot(1, 4, 2), plt.imshow(np.rot90(label.cpu().squeeze()), cmap='gray'), plt.title('ground truth')
            plt.subplot(1, 4, 3), plt.imshow(np.rot90(output_down_real.cpu().squeeze()), cmap='gray'), plt.title('recon_real')
            plt.subplot(1, 4, 4), plt.imshow(np.rot90(ot), cmap='gray'), plt.title('recon_all')'''
            #####  重建结果输出  #####
            '''plt.figure(20)  # 真值image
            plt.imshow(np.rot90(label.cpu().squeeze()), cmap='gray')   #, plt.title("groundtruth")
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            # plt.savefig("rec/GT0.2.eps", bbox_inches='tight', pad_inches=0)
            plt.figure(21)  # 重建image
            plt.imshow(np.rot90(output_down_real.cpu().squeeze()), cmap='gray')   #, plt.title("reconstruction")
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.savefig("rec/no_MLRF0.2.png", bbox_inches='tight', pad_inches=0)'''
            # plt.show()
            '''# batch size >= 2
            plt.subplot(1, 4, 1), plt.imshow(np.rot90(net_img_down_0.cpu().squeeze()[0, :, :]), cmap='gray'), plt.title('input')
            plt.subplot(1, 4, 2), plt.imshow(np.rot90(label.cpu().squeeze()[0, :, :]), cmap='gray'), plt.title('ground truth')
            plt.subplot(1, 4, 3), plt.imshow(np.rot90(output_down_real.cpu().squeeze()[0, :, :]), cmap='gray'), plt.title('recon_real')
            plt.subplot(1, 4, 4), plt.imshow(np.rot90(ot), cmap='gray'), plt.title('recon_all')

            plt.figure(2)  # error map
            # error = np.rot90(output_down_real.cpu().squeeze() - label.cpu().squeeze())  # batch size = 1
            error = np.rot90(output_down_real.cpu().squeeze()[0, :, :] - label.cpu().squeeze()[0, :, :])  # 旋转90°
            norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
            plt.imshow(error, cmap='bwr', norm=norm), plt.title('error map')
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.colorbar()
            plt.show()'''

            '''plt.figure(2)  # error map batch size = 1
            error = np.rot90(abs(output_down_real.cpu().squeeze() - label.cpu().squeeze()))  # 旋转90°
            norm = matplotlib.colors.Normalize()
            plt.imshow(error, cmap='rainbow', norm=norm)  #, plt.title('error map')
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            # plt.savefig("model_img0.4/all_error_30.eps", bbox_inches='tight', pad_inches=0)
            plt.colorbar()

            # sns error map
            error = np.rot90(output_down_real.cpu().squeeze() - label.cpu().squeeze())  # 旋转90°
            # error = cv2.GaussianBlur(error, (3, 3), 0) * 2
            error = sns.heatmap(data=error, center=0, vmin=-0.08, vmax=0.08, cmap='bwr')
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.show()'''

            '''plt.figure(2)  # error map
            # error = np.rot90(output_down_real.cpu().squeeze() - label.cpu().squeeze())  # batch size = 1
            error = np.rot90(label.cpu().squeeze() - ((output_up_real + output_down_real)/2).cpu().squeeze())  # 旋转90°
            norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1)
            plt.imshow(error, cmap='bwr', norm=norm)# , plt.title('error map')
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.savefig("rec/error_no_MLRF0.2.png", bbox_inches='tight', pad_inches=0)
            plt.colorbar()'''
            # plt.show()

            '''plt.figure(1)
            plt.imshow(np.rot90(label.cpu().squeeze()), cmap='gray'), plt.axis('off'), plt.xticks([]), plt.yticks([])
            # plt.savefig("figs/label02.png", bbox_inches='tight', pad_inches=0)  # plt.title('label')
            plt.figure(2)
            plt.imshow(np.rot90(refer.cpu().squeeze()), cmap='gray'), plt.axis('off'), plt.xticks([]), plt.yticks([])
            # plt.savefig("result_image/refer.eps", bbox_inches='tight', pad_inches=0, dpi=1200)  # plt.title('refer')

            plt.figure(31)  # 重建image
            plt.imshow(np.rot90(output_down_real.cpu().squeeze()), cmap='gray')  # , plt.title("reconstruction")
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.savefig("figs/result_PD_SANet_random.png", bbox_inches='tight', pad_inches=0)

            plt.figure(32)  # error map
            # error = np.rot90(output_down_real.cpu().squeeze() - label.cpu().squeeze())  # batch size = 1
            error = np.rot90(label.cpu().squeeze() - output_down_real.cpu().squeeze())  # 旋转90°
            norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
            plt.imshow(error, cmap='bwr', norm=norm) #, plt.title('error map')
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            plt.savefig("figs/error_PD_SANet_random.png", bbox_inches='tight', pad_inches=0)
            plt.colorbar()
            plt.show()'''

            #######PJS数据集绘图########
            '''la = 1 + np.abs(label[0, :, :].cpu())
            plt.figure(1)
            plt.imshow(np.rot90(la), cmap='gray')   #, plt.title('IXI_T2')
            plt.axis('off'), plt.xticks([]), plt.yticks([])

            plt.figure(31)  # 重建image
            plt.imshow(np.rot90(ot), cmap='gray')  # , plt.title("reconstruction")
            plt.axis('off'), plt.xticks([]), plt.yticks([])
            # plt.savefig("data/Our_PD_random0.4.png", bbox_inches='tight', pad_inches=0)
            plt.show()'''



        # 边缘检测 [1, 1, 256, 256]
        edge_up = sobel_plus(output_up_image.permute(0, 3, 1, 2).contiguous())
        edge_down = sobel_plus(output_down_image.permute(0, 3, 1, 2).contiguous())

        '''plt.figure(3)   # sobel边缘检测结果
        plt.subplot(1, 2, 1), plt.imshow(np.rot90(edge_up.cpu().detach().numpy().squeeze()), cmap='gray'), plt.title('edge_up')
        plt.subplot(1, 2, 2), plt.imshow(np.rot90(edge_down.cpu().detach().numpy().squeeze()), cmap='gray'), plt.title('edge_down')
        plt.show()'''

        # 损失计算
        diff_otherf = (output_up_kspace - output_down_kspace) * (1 - mask_under)
        # diff_image = (output_up_image - output_down_image)    # 原始
        diff_image = (output_up_image - output_down_image) * (1 - mask_under)
        recon_loss_up = criterion(output_up_kspace * mask_under, under_kspace)  # 重建损失up
        recon_loss_down = criterion(output_down_kspace * mask_under, under_kspace)  # 重建损失down
        diff_loss = criterion(diff_otherf, torch.zeros_like(diff_otherf))  # 一致性损失（两结果之差趋近于0）
        diff_loss_img = criterion(diff_image, torch.zeros_like(diff_image))

        # edge_loss = criterion(edge_up, edge_down)
        edge_loss_0 = (edge_up.permute(0, 2, 3, 1) - edge_down.permute(0, 2, 3, 1)) * (1 - mask_under)
        edge_loss = criterion(edge_loss_0, torch.zeros_like(edge_loss_0))

        constr_loss_up = criterion(loss_layers_up[0], torch.zeros_like(loss_layers_up[0]))
        constr_loss_down = criterion(loss_layers_down[0], torch.zeros_like(loss_layers_down[0]))
        for i in range(args.num_layers - 1):
            constr_loss_up += criterion(loss_layers_up[i + 1], torch.zeros_like((loss_layers_up[i + 1])))
            constr_loss_down += criterion(loss_layers_down[i + 1], torch.zeros_like((loss_layers_down[i + 1])))
        batch_loss = 0.01 * edge_loss + recon_loss_up + recon_loss_down + 0.01 * diff_loss + 0.01 * diff_loss_img + 0.01 * constr_loss_up + 0.01 * constr_loss_down
        # batch_loss = recon_loss_up + recon_loss_down + 0.01 * diff_loss + 0.01 * diff_loss_img + 0.01 * constr_loss_up + 0.01 * constr_loss_down
        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            '''psnr_up += psnr_slice(label, output_down_real)
            ssim_up += ssim_slice(label, output_down_real)
            mse_up += mse_slice(label, output_down_real)

            psnr_down += psnr_slice(label_extend, output_down_image)
            ssim_down += ssim_slice(label_extend, output_down_image)
            mse_down += mse_slice(label_extend, output_down_image)'''

            psnr_up += psnr_slice(label, (output_up_real + output_down_real) / 2)
            # ssim_up += ssim_slice(label_extend, output_up_image)
            ssim_up += ssim_slice(label, (output_up_real + output_down_real) / 2)
            mse_up += mse_slice(label, (output_up_real + output_down_real) / 2)    # 原图vs生成真值

            psnr_down += psnr_slice(label_extend, (output_down_image + output_down_image) / 2)
            ssim_down += ssim_slice(label_extend, (output_down_image + output_down_image) / 2)
            mse_down += mse_slice(label_extend, (output_down_image + output_down_image) / 2)

        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr_up /= len(dataloader)
        ssim_up /= len(dataloader)
        mse_up /= len(dataloader)
        psnr_down /= len(dataloader)
        ssim_down /= len(dataloader)
        mse_down /= len(dataloader)
        log.append(psnr_up)
        log.append(ssim_up)
        log.append(mse_up)
        log.append(psnr_down)
        log.append(ssim_down)
        log.append(mse_down)
    return log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger()
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='gloo', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim = 0.0
    # model
    model = MRInet(num_layers=args.num_layers, rank=rank)

    # whether load checkpoint
    if args.pretrained or args.mode == 'test':
        model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        args.lr = lr
        best_ssim = checkpoint['best_ssim']
        model.load_state_dict(checkpoint['model'])
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim))
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        init_weights(model, init_type=args.init_type, gain=args.gain)
        if rank == 0:
            logger.info('Initialize model with {}.'.format(args.init_type))
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # criterion, optimizer, learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    '''
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    '''

    if not args.pretrained:
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
        scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=20)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)

    # test step
    if args.mode == 'test':
        test_set = Dataset(args.test_path, args.test1_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.test_sample_rate)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        if rank == 0:
            logger.info('The size of test dataset is {}.'.format(len(test_set)))
            logger.info('Now testing {}.'.format(args.exp_name))
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            test_log = forward('test', rank, model, test_loader, criterion, optimizer, test_log, args)
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr_up = test_log[1]
        test_ssim_up = test_log[2]
        test_mse_up = test_log[3]
        test_psnr_down = test_log[4]
        test_ssim_down = test_log[5]
        test_mse_down = test_log[6]
        if rank == 0:
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\tpsnr_up:{:.5f}\tssim_up:{:.5f}\tmse_up:{:.7f}\tpsnr_down:{:.5f}\tssim_down:{:.5f}\tmse_down:{:.7f}'.format(test_time,
                                                                                                                     test_loss, test_psnr_up, test_ssim_up, test_mse_up, test_psnr_down, test_ssim_down, test_mse_down))
        return

    # training step
    train_set = Dataset(args.train_path, args.train1_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.train_sample_rate)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler
    )
    val_set = Dataset(args.val_path, args.val1_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.val_sample_rate)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    if rank == 0:
        logger.info('The size of training dataset and validation dataset is {} and {}, respectively.'.format(len(train_set), len(val_set)))
        logger.info('Now training {}.'.format(args.exp_name))
        writer = SummaryWriter(args.loss_curve_path)
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train_log = [epoch]
        epoch_start_time = time.time()
        model.train()
        train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args)
        model.eval()
        with torch.no_grad():
            train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args)
        epoch_time = time.time() - epoch_start_time
        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = train_log[2]
        val_loss = train_log[3]
        val_psnr = train_log[4]
        val_ssim = train_log[5]
        val_mse = train_log[6]

        is_best = val_ssim > best_ssim
        best_ssim = max(val_ssim, best_ssim)
        if rank == 0:
            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}\tval_mse:{:.7f}'.format(epoch, epoch_time, lr, train_loss, val_loss, val_psnr, val_ssim, val_mse))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # save checkpoint
            checkpoint = {
                'epoch': epoch,
                'lr': lr,
                'best_ssim': best_ssim,
                'model': model.module.state_dict()
            }
            if not os.path.exists(args.model_save_path):
                os.makedirs(args.model_save_path)
            model_path = os.path.join(args.model_save_path, 'checkpoint.pth.tar')
            best_model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
            if is_best:
                shutil.copy(model_path, best_model_path)
        # scheduler
        if epoch <= args.warmup_epochs and not args.pretrained:
            scheduler_wu.step()
        scheduler_re.step(val_ssim)
        early_stopping(val_ssim, loss=False)
        if early_stopping.early_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
            break
    if rank == 0:
        writer.close()
    return


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == '__main__':
    main()
