import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from torch.utils.data.distributed import DistributedSampler


class Data:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, test=False):

        train_path = os.path.join(self.config.data.train_data_dir)
        val_path = os.path.join(self.config.data.test_data_dir)

        train_dataset = MyDataset(train_path,
                                  n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_size,
                                  transforms=self.transforms,
                                  parse_patches=parse_patches)
        val_dataset = MyDataset(val_path,
                                n=self.config.training.patch_n,
                                patch_size=self.config.data.image_size,
                                transforms=self.transforms,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        # 训练数据

        # 评估数据
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=True, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        if not test:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                       shuffle=False, sampler=DistributedSampler(train_dataset),
                                                       num_workers=self.config.data.num_workers,
                                                       prefetch_factor=2,
                                                       pin_memory=True)
            return train_loader, val_loader

        if test:
            return val_loader


# 数据集加载类
class MyDataset(torch.utils.data.Dataset):
    parse_patches: bool

    def __init__(self, dir, patch_size, n, transforms, parse_patches=True):
        super().__init__()

        self.dir = dir
        zhuan_shu = os.listdir(dir + 'zhuanshu')
        li_shu = os.listdir(dir + 'lishu')
        kai_shu = os.listdir(dir + 'kaishu')
        jia_gu = os.listdir(dir + 'jiagu')
        jin_wen = os.listdir(dir + 'jinwen')
        zhan_guo = os.listdir(dir + 'zhanguo')

        self.zhuan_shu = zhuan_shu
        self.li_shu = li_shu
        self.kai_shu = kai_shu
        self.jia_gu = jia_gu
        self.jin_wen = jin_wen
        self.zhan_guo = zhan_guo
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0], [0], h, w

        interval_h = 48
        interval_w = 48
        i_list = [i * interval_h for i in range(n)]
        j_list = [j * interval_w for j in range(n)]
        # print(i_list)
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            # new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            # crops.append(new_crop)
            for j in range(len(y)):
                new_crop = img.crop((y[j], x[i], y[j] + w, x[i] + h))
                crops.append(new_crop)
        # print(len(crops))
        return tuple(crops)

    @staticmethod
    def get_params_previous(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0], [0], h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]

        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops_previous(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
            # for j in range(len(y)):
            #     new_crop = img.crop((y[j], x[i], y[j] + w, x[i] + h))
            #     crops.append(new_crop)
        # print(len(crops))
        return tuple(crops)

    def get_images(self, index):
        zhuanshu_name = self.zhuan_shu[index]
        lishu_name = self.li_shu[index]
        kaishu_name = self.kai_shu[index]
        jiagu_name = self.jia_gu[index]
        jinwen_name = self.jin_wen[index]
        zhanguo_name = self.zhan_guo[index]
        img_id = re.split('/', zhuanshu_name)[-1][:-4]
        zhuanshu_img = PIL.Image.open(os.path.join(self.dir, 'zhuanshu', zhuanshu_name)).convert(
            'RGB') if self.dir else PIL.Image.open(zhuanshu_name)
        try:
            lishu_img = PIL.Image.open(os.path.join(self.dir, 'lishu', lishu_name)).convert(
                'RGB') if self.dir else PIL.Image.open(lishu_name)
        except:
            lishu_img = PIL.Image.open(os.path.join(self.dir, 'lishu', lishu_name)).convert('RGB') if self.dir else \
                PIL.Image.open(lishu_name).convert('RGB')
        try:
            kaishu_img = PIL.Image.open(os.path.join(self.dir, 'kaishu', kaishu_name)).convert(
                'RGB') if self.dir else PIL.Image.open(kaishu_name)
        except:
            kaishu_img = PIL.Image.open(os.path.join(self.dir, 'kaishu', kaishu_name)).convert('RGB') if self.dir else \
                PIL.Image.open(kaishu_name).convert('RGB')
        try:
            jiagu_img = PIL.Image.open(os.path.join(self.dir, 'jiagu', jiagu_name)).convert(
                'RGB') if self.dir else PIL.Image.open(jiagu_name)
        except:
            jiagu_img = PIL.Image.open(os.path.join(self.dir, 'jiagu', jiagu_name)).convert('RGB') if self.dir else \
                PIL.Image.open(jiagu_name).convert('RGB')
        try:
            jinwen_img = PIL.Image.open(os.path.join(self.dir, 'jinwen', jinwen_name)).convert(
                'RGB') if self.dir else PIL.Image.open(jinwen_name)
        except:
            jinwen_img = PIL.Image.open(os.path.join(self.dir, 'jinwen', jinwen_name)).convert('RGB') if self.dir else \
                PIL.Image.open(jinwen_name).convert('RGB')
        try:
            zhanguo_img = PIL.Image.open(os.path.join(self.dir, 'zhanguo', zhanguo_name)).convert(
                'RGB') if self.dir else PIL.Image.open(zhanguo_name)
        except:
            zhanguo_img = PIL.Image.open(os.path.join(self.dir, 'zhanguo', zhanguo_name)).convert('RGB') if self.dir else \
                PIL.Image.open(zhanguo_name).convert('RGB')
        

        zhuanshu_img = zhuanshu_img.resize((112, 112), PIL.Image.LANCZOS)
        lishu_img = lishu_img.resize((112, 112), PIL.Image.LANCZOS)
        kaishu_img = kaishu_img.resize((112, 112), PIL.Image.LANCZOS)
        jiagu_img = jiagu_img.resize((112, 112), PIL.Image.LANCZOS)
        jinwen_img = jinwen_img.resize((112, 112), PIL.Image.LANCZOS)
        zhanguo_img = zhanguo_img.resize((112, 112), PIL.Image.LANCZOS)

        if self.parse_patches:
            # train时候是随即裁
            i, j, h, w = self.get_params(zhuanshu_img, (self.patch_size, self.patch_size), self.n)
            zhuanshu_imgs = self.n_random_crops(zhuanshu_img, i, j, h, w)
            lishu_imgs = self.n_random_crops(lishu_img, i, j, h, w)
            kaishu_imgs = self.n_random_crops(kaishu_img, i, j, h, w)
            jiagu_imgs = self.n_random_crops(jiagu_img, i, j, h, w)
            jinwen_imgs = self.n_random_crops(jinwen_img, i, j, h, w)
            zhanguo_imgs = self.n_random_crops(zhanguo_img, i, j, h, w)

            output = [torch.cat([self.transforms(kaishu_imgs[i]), self.transforms(zhuanshu_imgs[i])], dim=0)
                       for i in range(self.n * self.n)]

            # 前边是固定patch，后边是原本OBS随即采样
            i1, j1, h1, w1 = self.get_params_previous(zhuanshu_img, (self.patch_size, self.patch_size), 8)
            zhuanshu_imgs1 = self.n_random_crops_previous(zhuanshu_img, i1, j1, h1, w1)
            lishu_imgs1 = self.n_random_crops_previous(lishu_img, i1, j1, h1, w1)
            kaishu_imgs1 = self.n_random_crops_previous(kaishu_img, i1, j1, h1, w1)
            jiagu_imgs1 = self.n_random_crops_previous(jiagu_img, i1, j1, h1, w1)
            jinwen_imgs1 = self.n_random_crops_previous(jinwen_img, i1, j1, h1, w1)
            zhanguo_imgs1 = self.n_random_crops_previous(zhanguo_img, i1, j1, h1, w1)

            output1 = [torch.cat([self.transforms(kaishu_imgs1[i]), self.transforms(zhuanshu_imgs1[i])], dim=0)
                       for i in range(8)]
            outputs = output1 + output + output + output
            # print(len(outputs)) # 17

            return torch.stack(outputs, dim=0), img_id
        else:
            wd_new, ht_new = zhuanshu_img.size
            zhuanshu_img = zhuanshu_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            lishu_img = lishu_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            kaishu_img = kaishu_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            jiagu_img = jiagu_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            jinwen_img = jinwen_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            zhanguo_img = zhanguo_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)

            return torch.cat([self.transforms(kaishu_img), self.transforms(zhuanshu_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.zhuan_shu)
