from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import numpy as np
import glob

def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, args, transforms=default_transform):
        self.args = args
        self.root_dir = args.root_dir
        self.transforms = transforms
        self.patch_idx = 0
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.patch_idx)+".npy")

        self.sid_list = []
        for item in glob.glob(self.args.root_dir+"patch/"+"*_patch.npy"):
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()
        assert len(self.sid_list) == self.patch_data.shape[0]
        # point to your patch location file
        self.patch_loc = np.load("/pghbio/dbmi/batmanlab/lisun/copd/gnn_shared/data/patch_data_32_6_reg/19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = self.patch_loc / self.patch_loc.max(0)

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        self.sid_list_len = len(self.sid_list)
        print(stage+" dataset size:", self.sid_list_len)

    def set_patch_idx(self, idx):
        self.patch_idx = idx
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(idx)+".npy")

    def __len__(self):
        return self.sid_list_len*self.args.num_patch

    def __getitem__(self, idx):
        idx = idx % self.sid_list_len
        img = self.patch_data[idx,:,:,:]
        img = img + 1024.
        img = self.transforms(img[None,:,:,:])
        img[0] = img[0]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
        img[1] = img[1]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        patch_loc_idx = self.patch_loc[self.patch_idx,:] # patch location
        adj = np.array([]) # adj matrix not needed for patch-level training

        return img, patch_loc_idx, adj
