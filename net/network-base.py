import matplotlib.pyplot as plt
import torch

from net.net_parts import *
from mri_tools import ifftshift, fftshift
import numpy as np
from hyattention import *
from scipy.spatial.distance import cdist
import matplotlib


def to_complex(data):   # k-space数据
    # from data[1,256,256,2] to [1,256,256]complex
    data = data[:, :, :, 0] + 1j * data[:, :, :, 1]
    return data


class ISTANetPlus(nn.Module):
    def __init__(self, num_layers, rank):
        super(ISTANetPlus, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(BasicBlock(self.rank))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, under_img, mask):
        x = under_img
        layers_sym = []
        for i in range(self.num_layers):
            [x, layer_sym] = self.layers[i](x, under_img, mask)
            layers_sym.append(layer_sym)
        x_final = x
        return [x_final, layers_sym]


class ISTANetPlus_1(nn.Module):
    def __init__(self, num_layers, rank):
        super(ISTANetPlus_1, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(BasicBlock_1(self.rank))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, under_img, mask):
        x = under_img
        layers_sym = []
        for i in range(self.num_layers):
            [x, layer_sym] = self.layers[i](x, under_img, mask)
            layers_sym.append(layer_sym)
        x_final = x
        return [x_final, layers_sym]


class ISTANetPlus_k(nn.Module):
    def __init__(self, num_layers, rank):
        super(ISTANetPlus_k, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(BasicBlock_k(self.rank))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, under_kspace, mask):
        k = under_kspace
        layers_sym = []
        for i in range(self.num_layers):
            [k, layer_sym] = self.layers[i](k, under_kspace, mask)
            layers_sym.append(layer_sym)
        k_final = k
        return [k_final, layers_sym]


def fft(input):
    # (N, 2, W, H) -> (N, W, H, 2)
    input = input.permute(0, 2, 3, 1)
    input = input[:, :, :, 0] + 1j * input[:, :, :, 1]
    input = torch.fft.fftn(input, dim=(-2, -1), norm='ortho')
    input = torch.view_as_real(input)
    input = fftshift(input, axes=(-3, -2))  # 将图像中的低频部分移动到图像的中心
    # (N, W, H, 2) -> (N, 2, W, H)
    input = input.permute(0, 3, 1, 2)
    return input


def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = ifftshift(input, axes=(-3, -2))
    input = torch.view_as_complex(input)
    input = torch.fft.ifftn(input, dim=(-2, -1), norm='ortho')
    input = torch.view_as_real(input)
    input = input.permute(0, 3, 1, 2)
    return input


class DC(nn.Module):
    def __init__(self):
        super(DC, self).__init__()
        self.w1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)

    def forward(self, rec, start, mask, is_img=False):
        if is_img:
            rec = fft(rec)
            start = fft(start)
        result = mask.permute(0, 3, 1, 2).contiguous() * (rec * self.w1 / (1 + self.w1) + start * 1 / (self.w1 + 1))
        result = result + (1 - mask.permute(0, 3, 1, 2).contiguous()) * rec
        if is_img:
            result = ifft(result)
        return result


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.w2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)

    def forward(self, x1, x2):
        return x1 * 1 / (1 + self.w2) + x2 * self.w2 / (self.w2 + 1)


# ① 多模态融合
class CAC(nn.Module):   # cut and change: Fourier transform
    def __init__(self):
        super(CAC, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, target, refer, is_img=False):
        refer_kspace = fft(refer)
        '''plt.figure(10)  # 初始输入配对参考模态kspace
        plt.imshow(np.rot90(np.log(1 + np.abs(to_complex(refer_kspace.permute(0, 2, 3, 1).contiguous().cpu().detach().numpy()).squeeze()))),
                   cmap='gray')
        # plt.title("refer kspace")
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.savefig("save/refer kspace.png", bbox_inches='tight', pad_inches=0)
        plt.show()'''

        if is_img:
            target = fft(target)
        a = 0.35  # a-(0, 1) 低频区域边长占比
        w1 = int((1 - a) * 128)
        w2 = int((1 + a) * 128)
        h1 = int((1 - a) * 128)
        h2 = int((1 + a) * 128)
        target_cut = target[:, :, w1: w2, h1: h2].cpu()  # 目标模态的中间低频区域
        target_cut = np.pad(target_cut, pad_width=((0, 0), (0, 0), (w1, 256-w2), (h1, 256-h2)), mode="constant", constant_values=(0, 0))
        target_cut = torch.from_numpy(target_cut).cuda()  # 转换为Tensor
        refer_else = refer_kspace[:, :, w1: w2, h1: h2].cpu()
        refer_else = np.pad(refer_else, pad_width=((0, 0), (0, 0), (w1, 256-w2), (h1, 256-h2)), mode="constant", constant_values=(0, 0))
        refer_else = torch.from_numpy(refer_else).cuda()  # 转换为Tensor
        refer_cut = refer_kspace - refer_else  # 参考模态的外侧高频区域
        change = target_cut + refer_cut
        if is_img:
            change = ifft(change)
        change = self.conv_1(change)
        change = self.conv_2(change)
        return change


class Octconv(nn.Module):
    def __init__(self):
        super(Octconv, self).__init__()
        # 首层OctaveConv卷积: H-H & H-L
        self.FOCconv_re = FirstOctaveConv(kernel_size=(3, 3), in_channels=2, out_channels=128).cuda()
        self.FOCconv_tg = FirstOctaveConv(kernel_size=(3, 3), in_channels=2, out_channels=384).cuda()
        # 第二层OctaveConv卷积: H-H & H-L & L-H & L-L
        self.OCconv = OctaveConv(kernel_size=(3, 3), in_channels=256, out_channels=8, bias=False, stride=2, alpha=0.75).cuda()
        self.res_down1 = nn.Conv2d(in_channels=6, out_channels=2, kernel_size=1).cuda()    # 1*1卷积
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, image, refer_image, is_kspace=False):
        if is_kspace:
            image = ifft(image)
        high_refer, low_refer = self.FOCconv_re(refer_image)  # high_refer(1, 64, 256, 256)
        high_img, low_img = self.FOCconv_tg(image)  # low_img(1, 192, 128, 128)

        i = high_refer, low_img
        high_out, low_out = self.OCconv(i)  # high(1, 2, 128, 128) low(1, 6, 64, 64)
        low_out = self.res_down1(low_out)   # (1, 2, 64, 64)
        # low_out = torch.narrow(low_out, 1, 0, 2)    # (1, 2, 64, 64)
        low_out = self.upsample(low_out)    # 上采样
        fus_result = high_out + low_out
        fus_result = self.upsample(fus_result)
        if is_kspace:
            fus_result = fft(fus_result)
        return fus_result


def Distances(a, b):   # 欧式距离 -> 相似度矩阵
    dist = cdist(a, b, metric='euclidean')  # 像素级欧氏距离[H, W]
    sim = 1 / (1 + dist)
    return sim  # 相似度矩阵


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        output = self.layers1(x)
        return output


class MRIReconNet(nn.Module):
    def __init__(self, num_layers, rank):
        super(MRIReconNet, self).__init__()
        self.num_layers = num_layers
        self.rank = rank

        # ① 多模态融合
        self.conv1_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        self.cut_change_up = CAC()
        self.cut_change_down = CAC()
        self.oct_fusion_kspace = Octconv()
        self.oct_fusion_img = Octconv()
        self.w3 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        self.w4 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        ################################################################################

        self.cnn1_up = ISTANetPlus_k(self.num_layers, self.rank)
        self.cnn1_down = ISTANetPlus_1(self.num_layers, self.rank)
        self.dc11 = DC()
        self.dc12 = DC()
        self.fusion11 = Fusion()
        self.fusion12 = Fusion()

        self.cnn2_up = ISTANetPlus_k(self.num_layers, self.rank)
        self.cnn2_down = ISTANetPlus_1(self.num_layers, self.rank)
        self.dc21 = DC()
        self.dc22 = DC()
        self.fusion21 = Fusion()
        self.fusion22 = Fusion()

        self.cnn3_up = ISTANetPlus_k(self.num_layers, self.rank)
        self.cnn3_down = ISTANetPlus_1(self.num_layers, self.rank)
        self.dc31 = DC()
        self.dc32 = DC()
        self.fusion31 = Fusion()
        self.fusion32 = Fusion()

        self.cnn4_up = ISTANetPlus_k(self.num_layers, self.rank)
        self.cnn4_down = ISTANetPlus_1(self.num_layers, self.rank)
        self.dc41 = DC()
        self.dc42 = DC()
        self.fusion41 = Fusion()
        self.fusion42 = Fusion()

        self.cnn5_up = ISTANetPlus_k(self.num_layers, self.rank)
        self.cnn5_down = ISTANetPlus_1(self.num_layers, self.rank)
        self.dc51 = DC()
        self.dc52 = DC()
        self.fusion51 = Fusion()
        self.fusion52 = Fusion()

        self.cnn6_up = ISTANetPlus_k(self.num_layers, self.rank)
        self.cnn6_down = ISTANetPlus_1(self.num_layers, self.rank)
        self.dc61 = DC()
        self.dc62 = DC()
        self.fusion61 = Fusion()
        self.fusion62 = Fusion()

        '''self.cnn7_up = ISTANetPlus_k(self.num_layers, self.rank)
        self.cnn7_down = ISTANetPlus_1(self.num_layers, self.rank)
        self.dc71 = DC()
        self.dc72 = DC()
        self.fusion71 = Fusion()
        self.fusion72 = Fusion()'''

        self.cat_fus = FusionNet()
        self.attention = AttentionBlock()

        # self.conv1_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, under_kspace_up, mask_up, under_img_down, mask_down, refer):
        ##### cut and change #####  # [B, 2, 256, 256]
        under_kspace_up_change = self.cut_change_up(under_kspace_up, refer)  # 上分支-k空间cac kspace
        under_img_down_change = self.cut_change_down(under_img_down, refer, True)  # 下分支-图像域cac img

        refer_image = refer[:, 0, :, :] + 1j * refer[:, 1, :, :]
        real = under_img_down[:, 0, :, :] + 1j * under_img_down[:, 1, :, :]
        real_change = under_img_down_change[:, 0, :, :] + 1j * under_img_down_change[:, 1, :, :]
        refer_image = refer_image.float().unsqueeze(-1)
        real = real.float().unsqueeze(-1)
        real_change = real_change.float().unsqueeze(-1)
        '''plt.figure(12)  # 参考 融合前/后
        plt.imshow(np.rot90(refer_image.cpu().squeeze()), cmap='gray'),
        plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.savefig("result_image/refer.png", bbox_inches='tight', pad_inches=0)  # plt.title('refer')
        plt.figure(13)
        plt.imshow(np.rot90(real.cpu().squeeze()), cmap='gray'),
        plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.savefig("result_image/TW2_before.png", bbox_inches='tight', pad_inches=0)  # plt.title('before')
        plt.figure(14)
        plt.imshow(np.rot90(real_change.cpu().detach().numpy().squeeze()), cmap='gray'),
        plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.savefig("result_image/TW2_10_0.05.eps", bbox_inches='tight', pad_inches=0)  # plt.title('after')
        plt.show()'''

        ##### octconv #####
        fus_kspace = self.oct_fusion_kspace(under_kspace_up, refer, True)  # 上分支-k空间oct kspace
        fus_img = self.oct_fusion_img(under_img_down, refer)  # 下分支-图像域oct image

        '''fus = fus_img[:, 0, :, :] + 1j * fus_img[:, 1, :, :]
        fus = fus.float().unsqueeze(-1)
        plt.figure(15)
        plt.imshow(np.rot90(fus.cpu().detach().numpy().squeeze()), cmap='gray'),
        plt.axis('off'), plt.xticks([]), plt.yticks([]), plt.savefig("save/oct fusion.png", bbox_inches='tight', pad_inches=0)
        # plt.title('oct fusion')
        plt.show()'''

        ##### 基于相似度注意权重的特征融合 #####
        cac_img_d = self.conv1_1(under_img_down_change[0, :, :, :]).squeeze()
        oct_img_d = self.conv1_1(fus_img[0, :, :, :]).squeeze()
        img_sim = Distances(cac_img_d.cpu().detach().numpy(), oct_img_d.cpu().detach().numpy())
        cac_k_u = self.conv1_1(under_kspace_up_change[0, :, :, :]).squeeze()
        oct_k_u = self.conv1_1(fus_kspace[0, :, :, :]).squeeze()
        k_sim = Distances(cac_k_u.cpu().detach().numpy(), oct_k_u.cpu().detach().numpy())
        img_sim = torch.from_numpy(img_sim).unsqueeze(0).unsqueeze(0).cuda()
        k_sim = torch.from_numpy(k_sim).unsqueeze(0).unsqueeze(0).cuda()
        re_img_sim = torch.as_tensor(img_sim, dtype=torch.float32)
        re_k_sim = torch.as_tensor(k_sim, dtype=torch.float32)
        '''plt.figure(16)
        plt.imshow(np.rot90(re_img_sim.cpu().detach().numpy().squeeze())),
        plt.axis('off'), plt.xticks([]), plt.yticks([])#, plt.savefig("save/img similarity.png", bbox_inches='tight', pad_inches=0)
        # plt.title('img similarity')
        plt.figure(17)
        plt.imshow(np.rot90(re_k_sim.cpu().detach().numpy().squeeze())),
        plt.axis('off'), plt.xticks([]), plt.yticks([])#, plt.savefig("save/kspace similarity.png", bbox_inches='tight', pad_inches=0)
        # plt.title('kspace similarity')
        plt.show()'''

        # up_x_1 = (under_kspace_up_change * 1 / (1 + self.w3) + fus_kspace * self.w3 / (1 + self.w3)) * (1 + re_k_sim)
        # down_x_1 = (under_img_down_change * 1 / (1 + self.w4) + fus_img * self.w4 / (1 + self.w4)) * (1 + re_img_sim)
        # u_x = up_x_1
        # d_x = down_x_1

        up_x_1 = (under_kspace_up_change * 1 / (1 + self.w3) + fus_kspace * self.w3 / (1 + self.w3)) * (1 + re_k_sim)
        down_x_1 = (under_img_down_change * 1 / (1 + self.w4) + fus_img * self.w4 / (1 + self.w4)) * (1 + re_img_sim)
        # u_x = up_x_1
        # d_x = down_x_1
        u_x = under_kspace_up
        d_x = under_img_down

        '''# 无多模态融合输入
        up_x_1_nF = under_kspace_up
        down_x_1_nF = under_img_down
        u_x = under_kspace_up
        d_x = under_img_down'''

        ##### first stage ##### # 上分支-k空间 下分支-图像域
        up_fea_1, loss_layers_up = self.cnn1_up(up_x_1, mask_up)     # 无多模态融合输入 up_x_1_nF / down_x_1_nF
        down_fea_1, loss_layers_down = self.cnn1_down(down_x_1, mask_down)

        rec_up_fea_1 = self.dc11(up_fea_1, u_x, mask_up)
        rec_down_fea_1 = self.dc12(down_fea_1, d_x, mask_down, True)

        up_to_down_1 = ifft(rec_up_fea_1)
        down_to_up_1 = fft(rec_down_fea_1)

        ##### second stage #####
        up_x_2 = self.fusion11(rec_up_fea_1, down_to_up_1)
        down_x_2 = self.fusion12(rec_down_fea_1, up_to_down_1)

        up_fea_2, loss_layers_up = self.cnn2_up(up_x_2, mask_up)
        down_fea_2, loss_layers_down = self.cnn2_down(down_x_2, mask_down)

        rec_up_fea_2 = self.dc21(up_fea_2, u_x, mask_up)
        rec_down_fea_2 = self.dc22(down_fea_2, d_x, mask_down, True)

        up_to_down_2 = ifft(rec_up_fea_2)
        down_to_up_2 = fft(rec_down_fea_2)

        ##### third stage #####
        up_x_3 = self.fusion21(rec_up_fea_2, down_to_up_2)
        down_x_3 = self.fusion22(rec_down_fea_2, up_to_down_2)

        up_fea_3, loss_layers_up = self.cnn3_up(up_x_3, mask_up)
        down_fea_3, loss_layers_down = self.cnn3_down(down_x_3, mask_down)

        rec_up_fea_3 = self.dc31(up_fea_3, u_x, mask_up)
        rec_down_fea_3 = self.dc32(down_fea_3, d_x, mask_down, True)

        up_to_down_3 = ifft(rec_up_fea_3)
        down_to_up_3 = fft(rec_down_fea_3)

        ##### forth Stage #####
        up_x_4 = self.fusion31(rec_up_fea_3, down_to_up_3)
        down_x_4 = self.fusion32(rec_down_fea_3, up_to_down_3)
        # output_up = self.fusion31(rec_up_fea_3, down_to_up_3)
        # output_down = self.fusion32(rec_down_fea_3, up_to_down_3)

        up_fea_4, loss_layers_up = self.cnn4_up(up_x_4, mask_up)
        down_fea_4, loss_layers_down = self.cnn4_down(down_x_4, mask_down)

        rec_up_fea_4 = self.dc41(up_fea_4, u_x, mask_up)
        rec_down_fea_4 = self.dc42(down_fea_4, d_x, mask_down, True)

        up_to_down_4 = ifft(rec_up_fea_4)
        down_to_up_4 = fft(rec_down_fea_4)

        output_up = self.fusion41(rec_up_fea_4, down_to_up_4)
        output_down = self.fusion42(rec_down_fea_4, up_to_down_4)

        ##### fifth Stage #####
        up_x_5 = self.fusion41(rec_up_fea_4, down_to_up_4)
        down_x_5 = self.fusion42(rec_down_fea_4, up_to_down_4)
        # output_up = self.fusion41(rec_up_fea_4, down_to_up_4)
        # output_down = self.fusion42(rec_down_fea_4, up_to_down_4)

        up_fea_5, loss_layers_up = self.cnn5_up(up_x_5, mask_up)
        down_fea_5, loss_layers_down = self.cnn5_down(down_x_5, mask_down)

        rec_up_fea_5 = self.dc51(up_fea_5, u_x, mask_up)
        rec_down_fea_5 = self.dc52(down_fea_5, d_x, mask_down, True)

        up_to_down_5 = ifft(rec_up_fea_5)
        down_to_up_5 = fft(rec_down_fea_5)

        # output_up = self.fusion51(rec_up_fea_5, down_to_up_5)
        # output_down = self.fusion52(rec_down_fea_5, up_to_down_5)

        ##### sixth Stage #####
        up_x_6 = self.fusion51(rec_up_fea_5, down_to_up_5)
        down_x_6 = self.fusion52(rec_down_fea_5, up_to_down_5)

        up_fea_6, loss_layers_up = self.cnn6_up(up_x_6, mask_up)
        down_fea_6, loss_layers_down = self.cnn6_down(down_x_6, mask_down)

        rec_up_fea_6 = self.dc61(up_fea_6, u_x, mask_up)
        rec_down_fea_6 = self.dc62(down_fea_6, d_x, mask_down, True)

        up_to_down_6 = ifft(rec_up_fea_6)
        down_to_up_6 = fft(rec_down_fea_6)

        output_up = self.fusion61(rec_up_fea_6, down_to_up_6)
        output_down = self.fusion62(rec_down_fea_6, up_to_down_6)

        '''##### seventh Stage #####
        up_x_7 = self.fusion71(rec_up_fea_6, down_to_up_6)
        down_x_7 = self.fusion72(rec_down_fea_6, up_to_down_6)

        up_fea_7, loss_layers_up = self.cnn7_up(up_x_7, mask_up)
        down_fea_7, loss_layers_down = self.cnn7_down(down_x_7, mask_down)

        rec_up_fea_7 = self.dc71(up_fea_7, u_x, mask_up)
        rec_down_fea_7 = self.dc72(down_fea_7, d_x, mask_down, True)

        up_to_down_7 = ifft(rec_up_fea_7)
        down_to_up_7 = fft(rec_down_fea_7)

        output_up = self.fusion71(rec_up_fea_7, down_to_up_7)
        output_down = self.fusion72(rec_down_fea_7, up_to_down_7)'''

        output_up1 = np.concatenate((up_x_2.cpu().detach().numpy(), up_x_3.cpu().detach().numpy(),
                                     up_x_4.cpu().detach().numpy(),  up_x_5.cpu().detach().numpy(),  up_x_6.cpu().detach().numpy(),
                                     # up_x_7.cpu().detach().numpy(),
                                     output_up.cpu().detach().numpy()), axis=1)
        output_down1 = np.concatenate((down_x_2.cpu().detach().numpy(), down_x_3.cpu().detach().numpy(),
                                       down_x_4.cpu().detach().numpy(),  down_x_5.cpu().detach().numpy(),  down_x_6.cpu().detach().numpy(),
                                       # down_x_7.cpu().detach().numpy(),
                                       output_down.cpu().detach().numpy()), axis=1)
        output_up2 = self.cat_fus(torch.from_numpy(output_up1).cuda())
        output_down2 = self.cat_fus(torch.from_numpy(output_down1).cuda())
        output_up = output_up + output_up2
        output_down = output_down + output_down2

        # hybrid attention
        output_up = self.attention(output_up)
        output_down = self.attention(output_down)

        return output_up, loss_layers_up, output_down, loss_layers_down
        # 1 2 256 256
