# extract features for downstream tasks

import os
import argparse
from easydict import EasyDict as edict
import json
import random
import numpy as np

import torch
import torch.nn as nn

from utils.cond_encoder import Encoder
from utils.gcn_eval import GraphNet
from dataset_test_full import COPD_dataset

from torch_geometric.data import Data, Batch
from torch_geometric.utils.sparse import dense_to_sparse

parser = argparse.ArgumentParser(description='Get 3D Images Patch Representation')
parser.add_argument('--exp-dir', help='path to work dir')
parser.add_argument('--root-dir', help='path to preprocessed data dir')
parser.add_argument('--checkpoint_patch', help='path to pretrained patch-level encoder checkpoint')
parser.add_argument('--checkpoint_graph', help='path to pretrained subject-level GCN checkpoint')
parser.add_argument('--batch-size', type=int, default=1)

def main():
    # read configurations
    p = parser.parse_args()
    with open(os.path.join(p.exp_dir, 'configs.json')) as f:
        args = edict(json.load(f))
    args.exp_dir = p.exp_dir
    args.checkpoint_patch = p.checkpoint_patch
    args.checkpoint_graph = p.checkpoint_graph
    args.batch_size = p.batch_size
    args.root_dir = p.root_dir

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # output path for extracted features
    os.makedirs(os.path.join(args.exp_dir, 'output_feature'), exist_ok=True)
    main_worker(args)

def main_worker(args):

    model = Encoder(args.moco_dim)
    model.fc = torch.nn.Sequential()
    state_dict = torch.load(args.checkpoint_patch)['state_dict']
    # preprocessing on ckpt names
    for k in list(state_dict.keys()):
        # retain only encoder_q
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
       # delete renamed or unused k
        del state_dict[k]

    # patch-level encoder
    model.load_state_dict(state_dict)
    print("CNN Weights loaded:", args.checkpoint_patch)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # GCN
    model2 = GraphNet(args.moco_dim)
    model2.fc = torch.nn.Sequential()
    
    state_dict = torch.load(args.checkpoint_graph)['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model2.load_state_dict(state_dict)
    print("GNN Weights loaded:", args.checkpoint_graph)
    model2 = model2.cuda()
    #model2 = torch.nn.DataParallel(model2).cuda()
    model2.eval()

    dataset = COPD_dataset("train", args)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=4, drop_last=False)

    pred_arr = np.empty((len(dataset), args.num_patch, args.moco_dim))

    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        images, patch_loc_idx, adj = batch
        images = images[0].float().cuda()
        patch_loc_idx = patch_loc_idx[0].float().cuda()
        adj = adj.cuda()
        with torch.no_grad():
            pred = model(images, patch_loc_idx)
            #if i == 0:
            #    print("\nCNN output shape:", pred.shape)
            batch = [Data(x=pred[:,:args.moco_dim], edge_index=dense_to_sparse(adj[0])[0])]
            batch = Batch.from_data_list(batch)
            batch.batch = batch.batch.cuda()
            pred = model2(batch)
            #if i == 0:
            #    print("GNN output shape:", pred.shape)
        pred_arr[i,:,:] = pred.cpu().numpy()
    
    # subject-level pooling
    pred_arr = np.mean(pred_arr, axis=1)

    # shape: [num_sample, feature_dimension]
    np.save(os.path.join(args.exp_dir, 'output_feature', "feature_array.npy"), pred_arr)
    print("\nFeature extraction finished.")

if __name__ == '__main__':
    main()
