import random
import pathlib
import scipy.io as sio
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import normalize_zero_to_one
import os

import torch.nn.functional as F


class IXIData(Dataset):
    def __init__(self, data_path, data_path_else, u_mask_path, s_mask_up_path, s_mask_down_path, sample_rate):
        super(IXIData, self).__init__()
        self.data_path = data_path
        self.data_path_else = data_path_else
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        self.sample_rate = sample_rate

        self.examples = []
        files = list(pathlib.Path(self.data_path).iterdir())
        # The middle slices have more detailed information, so it is more difficult to reconstruct.
        start_id, end_id = 66, 100   #PJS数据 130, 230
        for file in sorted(files):
            self.examples += [(file, slice_id) for slice_id in range(start_id, end_id)]
        if self.sample_rate < 1:
            # random.shuffle(self.examples)
            num_examples = round(len(self.examples) * self.sample_rate)
            self.examples = self.examples[:num_examples]

        self.refer = []
        files_refer = list(pathlib.Path(self.data_path_else).iterdir())  # 按照ASCII码顺序排列
        #files_refer = os.listdir(self.data_path_else)
        #files_refer.sort(key=lambda x: int(x.split('.nii')[0]))  # 按照数字顺序大小
        for file_refer in files_refer:
            self.refer += [(file_refer, slice_id_refer) for slice_id_refer in range(start_id, end_id)]
        self.refer.sort(key=lambda x: int("".join(filter(str.isdigit, os.path.basename(str(x))))))  # 按照数字顺序大小排列

        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down
        # self.mask_net_up = self.mask_under
        # self.mask_net_down = self.mask_under

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file, slice_id = self.examples[item]
        data = nib.load(file)   # 读入.nii格式图像
        label = data.dataobj[..., slice_id]
        label = normalize_zero_to_one(label, eps=1e-6)
        label = torch.from_numpy(label).unsqueeze(-1).float()

        filename = os.path.basename(file)
        file_id = int("".join(filter(str.isdigit, filename)))

        number = (file_id - 1) * 34 + slice_id - 66
        # number = (file_id - 1) * 34 + slice_id - 66
        # number = (file_id - 1) * 51 + slice_id - 49

        file_refer, slice_id_refer = self.refer[number]  # 获得对应匹配的参考模态切片

        data_refer = nib.load(file_refer)  # 读入.nii格式图像
        refer = data_refer.dataobj[..., slice_id_refer]
        refer = normalize_zero_to_one(refer, eps=1e-6)
        refer = torch.from_numpy(refer).unsqueeze(-1).float()

        '''refer = refer.permute(2, 0, 1).unsqueeze(0)  # 调整维度顺序，变成 (1, 1, 240, 240)
        label = label.permute(2, 0, 1).unsqueeze(0)  # 同样调整维度顺序

        # 使用 interpolate 放大到 (256, 256)
        refer_resized = F.interpolate(refer, size=(256, 256), mode='bilinear', align_corners=False)
        label_resized = F.interpolate(label, size=(256, 256), mode='bilinear', align_corners=False)

        # 还原为 (256, 256, 1) 形状
        refer = refer_resized.squeeze(0).permute(1, 2, 0)
        label = label_resized.squeeze(0).permute(1, 2, 0)'''

        return label, self.mask_under, self.mask_net_up, self.mask_net_down, file.name, slice_id, refer

