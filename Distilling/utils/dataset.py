import torch
import tifffile
import torch.utils.data as data
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import random


class SegmentationDataSet(data.Dataset):

    def __init__(self, root, list_path, transform_img=None, transform_label=None):
        self.root = root
        self.list_path = list_path
        self.list_ids = [i_id.strip() for i_id in open(list_path)]

        # Data augmentation
        assert bool(transform_img) == bool(transform_label)
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index: int):
        name = self.list_ids[index]
        img = Image.open(osp.join(self.root, "img/%s" % (name))).convert("RGB")
        # print(np.shape(img))
        label = Image.open(osp.join(self.root, "label/%s" %
                           (name))).convert("RGB")

        # Preprocessing à la main
        img_np = np.asarray(img)
        label_np = np.asarray(label)

        img_np = img_np.transpose((2, 0, 1))  # Channel Arrangement
        label_np = label_np.transpose((2, 0, 1))
        img_np = img_np/255  # NORMALIZATION
        label_np = label_np/255  # Should we normalize the mask ?

        # If a pixel of label is 1 then it is of class 1 else it is of class 0
        label_np[0] = np.where(label_np[0] == 1, 1, 0)
        label_np[1] = np.where(label_np[1] == 0, 1, 0)

        # Remove useless last dimension (not colors but classes now)
        label_np = np.delete(label_np, 2, 0)


        img_tensor = torch.as_tensor(img_np.copy()).float().contiguous()
        label_tensor = torch.as_tensor(label_np.copy()).float().contiguous()

        # Data augmentation
        if self.transform_img:
            # Forcer la transfo a être deux fois la meme
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            img_tensor = self.transform_img(img_tensor)
            torch.manual_seed(seed)
            label_tensor = self.transform_label(label_tensor)

        # print("size image :", np.shape(img_np))
        # print("size label :", np.shape(label_np))

        return [img_tensor,
            label_tensor]


class SegmentationMixDataSet(data.Dataset):

    def __init__(self, first_root, first_list_path, second_root, second_list_path):
        self.first_root = first_root
        self.first_list_path = first_list_path
        self.second_root = second_root
        self.second_list_path = second_list_path
        self.first_list_ids = [i_id.strip() for i_id in open(first_list_path)]
        self.second_list_ids = [i_id.strip()
                                for i_id in open(second_list_path)]

    def __len__(self):
        return len(self.first_list_ids) + len(self.second_list_ids)

    def __getitem__(self, index: int):

        if (index < len(self.first_list_ids)):

            name = self.first_list_ids[index]
            img = Image.open(osp.join(self.first_root, "img/%s" %
                             (name))).convert("RGB")
            # print(np.shape(img))
            label = Image.open(
                osp.join(self.first_root, "label/%s" % (name))).convert("RGB")

            # Preprocessing à la main
            img_np = np.asarray(img)
            label_np = np.asarray(label)

            img_np = img_np.transpose((2, 0, 1))  # Channel Arrangement
            label_np = label_np.transpose((2, 0, 1))
            img_np = img_np/255  # NORMALIZATION
            label_np = label_np/255  # Should we normalize the mask ?

        else:

            name = self.second_list_ids[index-len(self.first_list_ids)]
            img = Image.open(
                osp.join(self.second_root, "img/%s" % (name))).convert("RGB")
            # print(np.shape(img))
            label = Image.open(
                osp.join(self.second_root, "label/%s" % (name))).convert("RGB")

            # Preprocessing à la main
            img_np = np.asarray(img)
            label_np = np.asarray(label)

            img_np = img_np.transpose((2, 0, 1))  # Channel Arrangement
            label_np = label_np.transpose((2, 0, 1))
            img_np = img_np/255  # NORMALIZATION
            label_np = label_np/255  # Should we normalize the mask ?

        return {
            'image': torch.as_tensor(img_np.copy()).float().contiguous(),
            'mask': torch.as_tensor(label_np.copy()).float().contiguous()
        }


class ImageDataSet(data.Dataset):

    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.list_ids = [i_id.strip() for i_id in open(list_path)]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self,
                    index: int):

        name = self.list_ids[index]
        img = Image.open(osp.join(self.root, "img/%s" % (name))).convert("RGB")
        # print(np.shape(img))

        # Preprocessing à la main
        img_np = np.asarray(img)

        img_np = img_np.transpose((2, 0, 1))  # Channel Arrangement
        img_np = img_np/255  # NORMALIZATION

        # print("size image :", np.shape(img_np))

        return {
            'image': torch.as_tensor(img_np.copy()).float().contiguous(),
        }
