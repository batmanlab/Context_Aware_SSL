import os
import numpy as np
import glob
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser(description='Data Preprocessing')
parser.add_argument('--num_patch', type=int, default=581)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_jobs', type=int, default=28)
parser.add_argument('--root_dir', type=str, default="/pghbio/dbmi/batmanlab/lisun/copd/gnn_shared/data/patch_data_32_6_reg_mask/")
args = parser.parse_args()

NUM_PATCH = args.num_patch
BATCH = args.batch_size
NUM_JOBS = args.num_jobs
ROOT_DIR = args.root_dir

def batch_load(path, batch_idx):
    sub_patches = []
    img = np.load(path)
    for j in range(batch_idx*BATCH, (batch_idx+1)*BATCH):
        if j >= NUM_PATCH:
            continue
        sub_patches.append(img[j,:,:,:])
    return sub_patches

def main():

    os.makedirs(ROOT_DIR+"grouped_patch/", exist_ok=True)
    sid_list = []
    for item in glob.glob(ROOT_DIR+"patch/"+"*_patch.npy"):
        sid_list.append(item)
    sid_list.sort()
#    sid_list = sid_list[:2] # test only
#    print(sid_list)
#    exit()
    for batch_idx in range(NUM_PATCH//BATCH+1):
        print("Processing batch", batch_idx)
        batch_patches = Parallel(n_jobs=NUM_JOBS)(delayed(batch_load)(item, batch_idx) for item in sid_list)
        patches = []
        for i in range(BATCH):
            if batch_idx*BATCH+i >= NUM_PATCH:
                continue
            patches.append([])
        for i in range(BATCH):
            if batch_idx*BATCH+i >= NUM_PATCH:
                continue
            for j in range(len(sid_list)):
                patches[i].append(batch_patches[j][i].copy())
                batch_patches[j][i] = None
        for i in range(BATCH):
            if batch_idx*BATCH+i >= NUM_PATCH:
                continue
            stack_patch = np.stack(patches[i])
            nan_mask = np.isnan(stack_patch) # Remove NaN
            stack_patch[nan_mask] = -1024
            np.save(ROOT_DIR+"grouped_patch/patch_loc_"+str(batch_idx*BATCH+i)+".npy", stack_patch)
        
if __name__ == '__main__':
    main()
