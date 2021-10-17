from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import numpy as np
import glob

def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, cfg, transforms=default_transform, graph_cutoff=0.13):
        self.cfg = cfg
        self.root_dir = cfg.root_dir
        self.metric_dict = dict() # initialize metric dictionary
        self.transforms = transforms
        self.graph_cutoff = graph_cutoff # hyper-parameter used to control graph sparsity
        
        # Read feature database for evaluation purpose only, replace it with your database or remove it
        FILE = open("/pghbio/dbmi/batmanlab/Data/COPDGene/ClinicalData/phase 1 Final 10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
        mylist = FILE.readline().strip("\n").split("\t")
        metric_idx = [mylist.index(label) for label in self.cfg.label_name]
        race_idx = mylist.index("race")
        for line in FILE.readlines():
            mylist = line.strip("\n").split("\t")
            tmp = [mylist[idx] for idx in metric_idx]
            if "" in tmp:
                continue
            metric_list = []
            for i in range(len(metric_idx)):
                metric_list.append(float(tmp[i]))
            self.metric_dict[mylist[0]] = metric_list
        FILE.close()

        # Filter images that doesn't have associated features
        self.sid_list = []
        for item in glob.glob(self.cfg.root_dir+"patch/"+"*_patch.npy"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()
        self.patch_loc = np.load("/pghbio/dbmi/batmanlab/lisun/copd/gnn_shared/data/patch_data_32_6_reg/19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = self.patch_loc / self.patch_loc.max(0) # column-wise norm

        self.sid_list = np.asarray(self.sid_list)
        print(stage+" dataset size:", len(self))

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img = np.load(self.root_dir+"patch/"+self.sid_list[idx]+"_patch.npy")
        img = img + 1024.
        
        # Get 2 augmented images (positive pair)
        img = self.transforms(img)
        img[0] = img[0][:,None,:,:,:]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
        img[1] = img[1][:,None,:,:,:]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        adj = np.load(self.root_dir+"adj/"+self.sid_list[idx]+"_adj.npy")
        adj=(adj>self.graph_cutoff).astype(np.int)

        return img, self.patch_loc.copy(), adj
