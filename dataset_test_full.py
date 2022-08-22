import os
from torch.utils.data import Dataset
import numpy as np
import glob

def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, cfg, transforms=default_transform):
        self.cfg = cfg
        self.root_dir = cfg.root_dir
        self.transforms = transforms

        self.sid_list = []
        for item in glob.glob(self.cfg.root_dir+"patch/"+"*_patch.npy"):
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()

        # location of landmarks, defined in atlas space
        self.patch_loc = np.load(self.cfg.root_dir+"19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = self.patch_loc / self.patch_loc.max(0) # column-wise norm

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        print(stage+" dataset size:", len(self))

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        # load image
        img = np.load(self.root_dir+"patch/"+self.sid_list[idx]+"_patch.npy")
        img = img + 1024.
        img = self.transforms(img)
        img = img[:,None,:,:,:]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        # load adjacency matrix
        adj = np.load(self.root_dir+"adj/"+self.sid_list[idx]+"_adj.npy")
        # binarize the graph
        # 0.13: hyperparameter defined to control the density of graph
        adj=(adj>0.13).astype(np.int)
        # always use location of landsmarks in atlas space
        return img, self.patch_loc.copy(), adj