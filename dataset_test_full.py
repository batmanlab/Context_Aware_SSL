from torch.utils.data import Dataset
import numpy as np
import glob

DATA_DIR = "/pghbio/dbmi/batmanlab/Data/COPDGene/ClinicalData/"
def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, cfg, transforms=default_transform):
        self.cfg = cfg
        self.root_dir = cfg.root_dir
        self.metric_dict = dict() # initialize metric dictionary
        self.transforms = transforms

        self.label_name = self.cfg.label_name + self.cfg.label_name_set2
        FILE = open(DATA_DIR+"phase 1 Final 10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
        mylist = FILE.readline().strip("\n").split("\t")
        metric_idx = [mylist.index(label) for label in self.label_name]
        for line in FILE.readlines():
            mylist = line.strip("\n").split("\t")
            tmp = [mylist[idx] for idx in metric_idx]
            if "" in tmp[:3]:
                continue
            metric_list = []
            for i in range(len(metric_idx)):
                if tmp[i] == "":
                    metric_list.append(-1024)
                else:
                    metric_list.append(float(tmp[i]))
            self.metric_dict[mylist[0]] = metric_list + [-1024,-1024,-1024]
        FILE = open(DATA_DIR+"CT scan datasets/CT visual scoring/COPDGene_CT_Visual_20JUL17.txt", "r")
        mylist = FILE.readline().strip("\n").split("\t")
        metric_idx = [mylist.index(label) for label in self.cfg.visual_score]
        for line in FILE.readlines():
            mylist = line.strip("\n").split("\t")
            if mylist[0] not in self.metric_dict:
                continue
            tmp = [mylist[idx] for idx in metric_idx]
            metric_list = []
            for i in range(len(metric_idx)):
                metric_list.append(float(tmp[i]))
            self.metric_dict[mylist[0]][-len(self.cfg.visual_score)-len(self.cfg.P2_Pheno):-len(self.cfg.P2_Pheno)] = metric_list
        FILE.close()
        FILE = open(DATA_DIR+'P1-P2 First 5K Long Data/Subject-flattened- one row per subject/First5000_P1P2_Pheno_Flat24sep16.txt', 'r')
        mylist = FILE.readline().strip("\n").split("\t")
        metric_idx = [mylist.index(label) for label in self.cfg.P2_Pheno]
        for line in FILE.readlines():
            mylist = line.strip("\n").split("\t")
            if mylist[0] not in self.metric_dict:
                continue
            tmp = [mylist[idx] for idx in metric_idx]
            metric_list = []
            for i in range(len(metric_idx)):
                metric_list.append(float(tmp[i]))
            self.metric_dict[mylist[0]][-len(self.cfg.P2_Pheno):] = metric_list
        FILE.close()

        self.sid_list = []
        for item in glob.glob(self.cfg.root_dir+"patch/"+"*_patch.npy"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()
        self.patch_loc = np.load(self.cfg.root_dir+"19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = self.patch_loc / self.patch_loc.max(0) # column-wise norm

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        print(stage+" dataset size:", len(self))

    def __len__(self):
        return len(self.sid_list)


    def __getitem__(self, idx):
        img = np.load(self.root_dir+"patch/"+self.sid_list[idx]+"_patch.npy")
        img = img + 1024.
        img = self.transforms(img)
        img = img[:,None,:,:,:]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        key = self.sid_list[idx][:6]
        label = np.asarray(self.metric_dict[key]) # TODO: self.sid_list[idx][:6] extract sid from the first 6 letters

        adj = np.load(self.root_dir+"adj/"+self.sid_list[idx]+"_adj.npy")
        adj=(adj>0.13).astype(np.int)

        return img, self.patch_loc.copy(), adj, label
