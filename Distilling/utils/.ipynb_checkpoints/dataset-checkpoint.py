import torch
import tifffile
import torch.utils.data as data
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image



class SegmentationDataSet(data.Dataset):
    
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
        #print(np.shape(img))
        label = Image.open(osp.join(self.root, "label/%s" % (name))).convert("RGB")
        
        #Preprocessing à la main
        img_np = np.asarray(img)
        label_np = np.asarray(label)
        
        img_np = img_np.transpose((2,0,1))       #Channel Arrangement
        label_np = label_np.transpose((2,0,1))
        img_np = img_np/255                      #NORMALIZATION
        label_np = label_np/255                 #Should we normalize the mask ?
        
        #print("size image :", np.shape(img_np))
        #print("size label :", np.shape(label_np))
        
        return {
            'image': torch.as_tensor(img_np.copy()).float().contiguous(),
            'mask': torch.as_tensor(label_np.copy()).float().contiguous()
        }
    
class SegmentationMixDataSet(data.Dataset):
    
    def __init__(self, first_root, first_list_path, second_root, second_list_path):
            self.first_root = first_root
            self.first_list_path = first_list_path
            self.second_root = second_root
            self.second_list_path = second_list_path
            self.first_list_ids = [i_id.strip() for i_id in open(first_list_path)]
            self.second_list_ids = [i_id.strip() for i_id in open(second_list_path)]

    def __len__(self):
        return len(self.first_list_ids) + len(self.second_list_ids)


    def __getitem__(self,index: int):
        
        if (index < len(self.first_list_ids)) :
        
            name = self.first_list_ids[index]
            img = Image.open(osp.join(self.first_root, "img/%s" % (name))).convert("RGB")
            #print(np.shape(img))
            label = Image.open(osp.join(self.first_root, "label/%s" % (name))).convert("RGB")
        
            #Preprocessing à la main
            img_np = np.asarray(img)
            label_np = np.asarray(label)
        
            img_np = img_np.transpose((2,0,1))       #Channel Arrangement
            label_np = label_np.transpose((2,0,1))
            img_np = img_np/255                      #NORMALIZATION
            label_np = label_np/255                 #Should we normalize the mask ?
            
        else : 
            
            name = self.second_list_ids[index-len(self.first_list_ids)]
            img = Image.open(osp.join(self.second_root, "img/%s" % (name))).convert("RGB")
            #print(np.shape(img))
            label = Image.open(osp.join(self.second_root, "label/%s" % (name))).convert("RGB")
        
            #Preprocessing à la main
            img_np = np.asarray(img)
            label_np = np.asarray(label)
        
            img_np = img_np.transpose((2,0,1))       #Channel Arrangement
            label_np = label_np.transpose((2,0,1))
            img_np = img_np/255                      #NORMALIZATION
            label_np = label_np/255                 #Should we normalize the mask ?
        
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
        #print(np.shape(img))
        
        #Preprocessing à la main
        img_np = np.asarray(img)
        
        img_np = img_np.transpose((2,0,1))       #Channel Arrangement
        img_np = img_np/255                      #NORMALIZATION
        
        #print("size image :", np.shape(img_np))
        
        return {
            'image': torch.as_tensor(img_np.copy()).float().contiguous(),
        }