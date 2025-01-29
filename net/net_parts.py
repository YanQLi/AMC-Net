import torch
import torch.nn as nn
import torch.nn.functional as F
from mri_tools import rAtA, rAtA_1, rAtA_k


'''# MHSA自注意力替换3*3卷积
class MHSA(nn.Module):
    def __init__(self, n_dims, width=32, height=32, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        # 32 4 w*h = 256
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        # 32 4 w*h = 256
        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class SampleCnn(nn.Module):
    def __init__(self):
        super(SampleCnn, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=8, padding=(1, 1))
        self.layers = nn.Sequential(
            MHSA(n_dims=32),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            MHSA(n_dims=32)
        )
        self.upsamp = torch.nn.Upsample(scale_factor=8, mode='nearest')

    def forward(self, x):
        x0 = self.pool1(x)
        x1 = self.layers(x0)
        output = self.upsamp(x1)
        return output'''


'''# Attention-Augmented-Conv2d自注意力增强替换3*3卷积
class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 4], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))

        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q1 = q * dkh ** -0.5
        # q *= dkh ** -0.5
        flat_q = torch.reshape(q1, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q1, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class SampleCnn(nn.Module):
    def __init__(self):
        super(SampleCnn, self).__init__()
        self.layers = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            AugmentedConv(in_channels=32, out_channels=32, kernel_size=3, dk=40, dv=4, Nh=1,
                          relative=False, stride=4),
            nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            AugmentedConv(in_channels=32, out_channels=32, kernel_size=3, dk=40, dv=4, Nh=1,
                          relative=False, stride=4)
        )
        self.upsamp = torch.nn.Upsample(scale_factor=16, mode='nearest')

    def forward(self, x):
        output = self.layers(x)
        output = self.upsamp(output)
        return output'''


'''class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 2, 3, padding=1, bias=False))

    def forward(self, inputs):
        output = self.D(inputs)
        return output
    def forward(self, inputs):
        # unsqueeze升维 / reshape改变tensor的形状 / transpose维度转置
        inputs = torch.unsqueeze(torch.reshape(torch.transpose(inputs, 0, 1), [-1, 64, 64]), dim=1)
        output = self.D(inputs)
        output = torch.transpose(torch.unsqueeze(torch.reshape(torch.squeeze(output), [-1, 256, 256]), dim=1), 0, 1)
        return output'''


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 2, 3, padding=1, bias=False))

    def forward(self, inputs):
        output = self.D(inputs)
        return output


# 自注意力并联卷积-替换传统3*3Conv
class SAttConv(nn.Module):
    def __init__(self, dim, dk, dv):
        super(SAttConv, self).__init__()
        self.scale = dk ** -0.5
        self.q = nn.Conv2d(dim, dk, kernel_size=1)  # dim输入特征数 dk输出特征数
        self.k = nn.Conv2d(dim, dk, kernel_size=1)
        self.v = nn.Conv2d(dim, dv, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.out = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.w22 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        '''self.out = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'''

    def forward(self, x):
        q1 = self.q(x)   # [c, h, w]
        k1 = self.k(x)
        v1 = self.v(x)   # [c, h, w]
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        x_sa = attn1 @ v1
        x_conv = self.conv(x)
        x_all = x_sa * self.w22 / (1 + self.w22) + x_conv * 1 / (1 + self.w22)
        # x_all = self.out(torch.cat((x_conv, x_sa), dim=1))
        # x_all = x_sa + x_conv
        '''q1 = self.q(x)  # [c, h, w]
        k1 = self.k(x)
        v1 = self.v(x)  # [c, h, w]
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        x_sa = attn1 @ v1
        x_conv = self.conv(x)
        x_all = self.out(torch.cat((x_conv, x_sa), dim=1))'''
        return x_all


class SampleCnn(nn.Module):
    def __init__(self):
        super(SampleCnn, self).__init__()
        self.layers = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SAttConv(dim=32, dk=32, dv=32),
            nn.ReLU(),
            SAttConv(dim=32, dk=32, dv=32)
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class BasicBlock(nn.Module):
    def __init__(self, rank):
        super(BasicBlock, self).__init__()
        self.rank = rank
        self.lambda_step = nn.Parameter(torch.FloatTensor([0.5]).cuda(self.rank), requires_grad=True)
        self.soft_thr = nn.Parameter(torch.FloatTensor([0.01]).cuda(self.rank), requires_grad=True)

        self.conv_in = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.f = SampleCnn()
        self.ft = SampleCnn()
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, under_img, mask):
        x = x - self.lambda_step * rAtA(x.permute(0, 2, 3, 1).contiguous(), mask).permute(0, 3, 1, 2).contiguous()
        x = x + self.lambda_step * under_img
        x_input = x

        x_D = self.conv_in(x_input)
        x_forward = self.f(x_D)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x_backward = self.ft(x)
        x_G = self.conv_out(x_backward)
        x_pred = x_input + x_G

        x_D_est = self.ft(x_forward)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


class BasicBlock_1(nn.Module):
    def __init__(self, rank):
        super(BasicBlock_1, self).__init__()
        self.rank = rank
        self.lambda_step = nn.Parameter(torch.FloatTensor([0.5]).cuda(self.rank), requires_grad=True)
        self.soft_thr = nn.Parameter(torch.FloatTensor([0.01]).cuda(self.rank), requires_grad=True)

        # self.conv_in = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_in = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.f = SampleCnn()
        self.ft = SampleCnn()
        # self.conv_out = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # self.denoiser = Denoiser()
        # self.w13 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)

    def forward(self, x, under_img, mask):
        x = x - self.lambda_step * rAtA_1(x.permute(0, 2, 3, 1).contiguous(), mask).permute(0, 3, 1, 2).contiguous()
        x = x + self.lambda_step * under_img
        # x = torch.unsqueeze(torch.reshape(torch.transpose(x, 0, 1), [-1, 64, 64]), dim=1)   # 输入切块
        x_input = x

        x_D = self.conv_in(x_input)
        x_forward = self.f(x_D)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x_backward = self.ft(x)
        x_G = self.conv_out(x_backward)
        x_pred = x_input + x_G

        x_D_est = self.ft(x_forward)
        symloss = x_D_est - x_D

        # 切块复原成完整图像
        # x_pred = torch.transpose(torch.unsqueeze(torch.reshape(torch.squeeze(x_pred), [-1, 256, 256]), dim=1), 0, 1)
        # x_pred = torch.reshape(x_pred, [-1, 2, 256, 256])   # batch_size = 2

        # x_pred = x_pred - self.denoiser(x_pred) * self.w13   # 去块效应

        return [x_pred, symloss]


class BasicBlock_k(nn.Module):
    def __init__(self, rank):
        super(BasicBlock_k, self).__init__()
        self.rank = rank
        self.lambda_step = nn.Parameter(torch.FloatTensor([0.5]).cuda(self.rank), requires_grad=True)
        self.soft_thr = nn.Parameter(torch.FloatTensor([0.01]).cuda(self.rank), requires_grad=True)

        # self.conv_in = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_in = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.f = SampleCnn()
        self.ft = SampleCnn()
        # self.conv_out = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # self.denoiser = Denoiser()
        # self.w14 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)

    def forward(self, k, under_kspace, mask):
        k = k - self.lambda_step * rAtA_k(k.permute(0, 2, 3, 1).contiguous(), mask).permute(0, 3, 1, 2).contiguous()
        k = k + self.lambda_step * under_kspace
        # k = torch.unsqueeze(torch.reshape(torch.transpose(k, 0, 1), [-1, 64, 64]), dim=1)   # 32 1 64 64
        k_input = k

        k_D = self.conv_in(k_input)
        k_forward = self.f(k_D)
        k = torch.mul(torch.sign(k_forward), F.relu(torch.abs(k_forward) - self.soft_thr))
        k_backward = self.ft(k)
        k_G = self.conv_out(k_backward)
        k_pred = k_input + k_G

        # k_pred = torch.transpose(torch.unsqueeze(torch.reshape(torch.squeeze(k_pred), [-1, 256, 256]), dim=1), 0, 1)
        # k_pred = torch.reshape(k_pred, [-1, 2, 256, 256])   # batch_size = 2

        # k_pred = k_pred - self.denoiser(k_pred) * self.w14

        k_D_est = self.ft(k_forward)
        symloss = k_D_est - k_D

        return [k_pred, symloss]


# ① 多模态融合 - octConv
class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)
        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)
        return X_h, X_l


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

