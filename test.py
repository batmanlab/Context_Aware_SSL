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
parser.add_argument('--exp-dir', default='./exp/custom_encoder_fusion_full_v2/')
parser.add_argument('--checkpoint_patch', default='0002')
parser.add_argument('--checkpoint_graph', default='0024')
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
    args.root_dir = "/pghbio/dbmi/batmanlab/lisun/copd/gnn_shared/data/patch_data_32_6_reg/"

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.exp_dir+args.checkpoint_patch.split('.')[0]+"_"+args.checkpoint_graph.split('.')[0], exist_ok=True)
    main_worker(args)

def main_worker(args):

    model = Encoder(args.moco_dim)
    model.fc = torch.nn.Sequential()
    state_dict = torch.load(args.exp_dir+"checkpoint_patch_"+args.checkpoint_patch+".pth.tar")['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
       # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict)
    print("CNN Weights loaded:", args.exp_dir+"checkpoint_patch_"+args.checkpoint_patch+".pth.tar")
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    model2 = GraphNet(args.moco_dim)
    model2.fc = torch.nn.Sequential()
    #dim_mlp = model2.fc.weight.shape[1]
    #model2.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model2.fc)
    state_dict = torch.load(args.exp_dir+"checkpoint_graph_"+args.checkpoint_graph+".pth.tar")['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
       # delete renamed or unused k
        del state_dict[k]

    model2.load_state_dict(state_dict)
    print("GNN Weights loaded:", args.exp_dir+"checkpoint_graph_"+args.checkpoint_graph+".pth.tar")
    model2 = model2.cuda()
    #model2 = torch.nn.DataParallel(model2).cuda()
    model2.eval()

    train_dataset = COPD_dataset("train", args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=4, drop_last=False)

    args.label_name = args.label_name + args.label_name_set2
    # train dataset
    pred_arr = np.empty((len(train_dataset), args.num_patch, args.moco_dim))
    feature_arr = np.empty((len(train_dataset), len(args.label_name)+len(args.visual_score)+len(args.P2_Pheno)))

    for i, batch in enumerate(train_loader):
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        images, patch_loc_idx, adj, labels = batch
        images = images[0].float().cuda()
        patch_loc_idx = patch_loc_idx[0].float().cuda()
        adj = adj.cuda()
        with torch.no_grad():
            pred = model(images, patch_loc_idx)
            if i == 0:
                print("\nCNN output shape:", pred.shape)
            batch = [Data(x=pred[:,:args.moco_dim],edge_index=dense_to_sparse(adj[0])[0])]
            batch = Batch.from_data_list(batch)
            batch.batch = batch.batch.cuda()
            pred = model2(batch)
            if i == 0:
                print("GNN output shape:", pred.shape)
        pred_arr[i,:,:] = pred.cpu().numpy()
        feature_arr[i:i+1,:] = labels
    np.save(args.exp_dir+args.checkpoint_patch.split('.')[0]+"_"+args.checkpoint_graph.split('.')[0]+"/pred_arr_full.npy", pred_arr)
    np.save(args.exp_dir+args.checkpoint_patch.split('.')[0]+"_"+args.checkpoint_graph.split('.')[0]+"/feature_arr_full.npy", feature_arr)
    print("\nEvaluation on full set finished.")

if __name__ == '__main__':
    main()
