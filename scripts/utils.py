#!/usr/bin/env python
# encoding: utf-8

import os
# from selectors import EpollSelector
import time
# import math
import torch
# import pickle
import argparse
import random
import numpy as np
import os.path as osp

import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

from layers import *
from models import *
from preprocessing import *
from collections import defaultdict
from torch_sparse import SparseTensor, coalesce
from convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_geometric.utils import dropout_adj, degree, to_undirected, k_hop_subgraph, subgraph
from torch_geometric.data import Data
from parser_data import parser_data
from pgd import PGD_contrastive
from sklearn.metrics import f1_score


def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)

    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
        if args.cuda in [0, 1, 2, 3]:
            device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        (row, col), value = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)
        model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes,
                         nlayer=args.All_num_layers, nhead=args.heads,
                         V=V, E=E)
    #     Below we can add different model, such as HyperGCN and so on
    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # assert len(result) == 3
        assert len(result) == 6
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest ACC Train: {result[:, 0].max():.2f}')
            print(f'Highest ACC Valid: {result[:, 1].max():.2f}')
            print(f'Highest ACC Test: {result[:, 2].max():.2f}')
            print(f'Highest F1 Train: {result[:, 3].max():.2f}')
            print(f'Highest F1 Valid: {result[:, 4].max():.2f}')
            print(f'Highest F1 Test: {result[:, 5].max():.2f}')
            print(f'  Final Train ACC: {result[argmax, 0]:.2f}')
            print(f'   Final Val ACC: {result[argmax, 1]:.2f}')
            print(f'   Final Test ACC: {result[argmax, 2]:.2f}')
            print(f'  Final Train F1: {result[argmax, 3]:.2f}')
            print(f'   Final Val F1: {result[argmax, 4]:.2f}')
            print(f'   Final Test F1: {result[argmax, 5]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            best_epoch = []
            for r in result:
                index = np.argmax(r[:, 1])
                best_epoch.append(index)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2_acc = r[r[:, 1].argmax(), 0].item()
                test_acc = r[r[:, 1].argmax(), 2].item()
                train2_f1 = r[r[:, 1].argmax(), 3].item()
                test_f1 = r[r[:, 1].argmax(), 5].item()
                best_results.append((train1, valid, train2_acc, test_acc,train2_f1,test_f1))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            print("best epoch:", best_epoch)
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final ACC Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final ACC Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'  Final F1 Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 5]
            print(f'   Final F1 Test: {r.mean():.2f} ± {r.std():.2f}')
            return best_result[:, 1], best_result[:, 3],best_result[:, 5]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            #             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

    train_f1 = eval_f1(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_f1 = eval_f1(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_f1 = eval_f1(
        data.y[split_idx['test']], out[split_idx['test']])


    #     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])



    return train_acc, valid_acc, test_acc, train_f1, valid_f1, test_f1,train_loss, valid_loss, test_loss, out


@torch.no_grad()
def evaluate_finetune(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model.forward_finetune(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

    train_f1 = eval_f1(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_f1 = eval_f1(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_f1 = eval_f1(
        data.y[split_idx['test']], out[split_idx['test']])

    #     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_f1, valid_f1, test_f1, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    #     ipdb.set_trace()
    #     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)

def eval_f1(y_true,y_pred):

    return f1_score(y_true.detach().cpu(), torch.argmax(y_pred, dim=-1).detach().cpu(), average='macro')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Main part of the training ---
# # Part 0: Parse arguments

def permute_edges(data, aug_ratio, permute_self_edge):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    # if not permute_self_edge:
    permute_num = int((edge_num - node_num) * aug_ratio)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()

    if args.add_e:
        idx_add_1 = np.random.choice(node_num, permute_num)
        idx_add_2 = np.random.choice(int(data.num_hyperedges), permute_num)
        # idx_add_2 = np.random.choice(int(data.num_hyperedges[0].item()), permute_num)
        idx_add = np.stack((idx_add_1, idx_add_2), axis=0)
    edge2remove_index = np.where(edge_index[1] < data.num_hyperedges[0].item())[0]
    edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges[0].item())[0]
    # edge2remove_index = np.where(edge_index[1] < data.num_hyperedges)[0]
    # edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges)[0]

    try:

        edge_keep_index = np.random.choice(edge2remove_index, (edge_num - node_num) - permute_num, replace=False)

    except ValueError:

        edge_keep_index = np.random.choice(edge2remove_index, (edge_num - node_num) - permute_num, replace=True)

    edge_after_remove1 = edge_index[:, edge_keep_index]
    edge_after_remove2 = edge_index[:, edge2keep_index]
    if args.add_e:
        edge_index = np.concatenate((edge_after_remove1, edge_after_remove2,
                                     ), axis=1)
    else:
        # edge_index = edge_after_remove
        edge_index = np.concatenate((edge_after_remove1, edge_after_remove2), axis=1)
    data.edge_index = torch.tensor(edge_index)
    # return data,[i for i in range(node_num)]
    return data,sorted(set([i  for i in range(data.x.shape[0])])),sorted(set(edge_index_orig[1][torch.where((edge_index_orig[1]<node_num+data.num_hyperedges) & (edge_index_orig[1]>node_num-1))[0]].cpu().numpy()))



def permute_hyperedges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    # hyperedge_num = int(data.num_hyperedges[0].item())
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()
    edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
    edge_remove_index_dict = {ind: i for i, ind in enumerate(edge_remove_index)}

    edge_remove_index_all = [i for i, he in enumerate(edge_index[1]) if he in edge_remove_index_dict]
    # print(len(edge_remove_index_all), edge_num, len(edge_remove_index), aug_ratio, hyperedge_num)
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove

    data.edge_index = torch.tensor(edge_index)

    return data,sorted(set([i for i in range(data.x.shape[0])])),sorted(set(edge_index_orig[1][torch.where((edge_index_orig[1]<node_num+data.num_hyperedges) & (edge_index_orig[1]>node_num-1))[0]].cpu().numpy()))


def adapt(data, aug_ratio, aug):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    # hyperedge_num = int(data.num_hyperedges[0].item())
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        index[he].append(i)
    # edge
    edge_index_orig = copy.deepcopy(data.edge_index)
    drop_weights = degree_drop_weights(data.edge_index, hyperedge_num)
    edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=aug_ratio, threshold=0.7, h=hyperedge_num,
                                      index=index)

    # feature
    edge_index_ = data.edge_index
    node_deg = degree(edge_index_[0])
    feature_weights = feature_drop_weights(data.x, node_c=node_deg)
    x_1 = drop_feature_weighted(data.x, feature_weights, aug_ratio, threshold=0.7)
    if aug == "adapt_edge":
        data.edge_index = edge_index_1
    elif aug == "adapt_feat":
        data.x = x_1
    else:
        data.edge_index = edge_index_1
        data.x = x_1
    return data,sorted(set([i  for i in range(data.x.shape[0])])),sorted(set(edge_index_orig[1][torch.where((edge_index_orig[1]<node_num+data.num_hyperedges) & (edge_index_orig[1]>node_num-1))[0]].cpu().numpy()))

def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p

    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def degree_drop_weights(edge_index, h):
    edge_index_ = edge_index
    deg = degree(edge_index_[1])[:h]
    # deg_col = deg[edge_index[1]].to(torch.float32)
    deg_col = deg
    s_col = torch.log(deg_col)
    # weights = (s_col.max() - s_col+1e-9) / (s_col.max() - s_col.mean()+1e-9)
    weights = (s_col - s_col.min() + 1e-9) / (s_col.mean() - s_col.min() + 1e-9)
    return weights


def feature_drop_weights(x, node_c):
    # x = x.to(torch.bool).to(torch.float32)
    x = torch.abs(x).to(torch.float32)
    # 100 x 2012 mat 2012-> 100
    w = x.t() @ node_c
    w = w.log() + 1e-7
    # s = (w.max() - w) / (w.max() - w.mean())
    s = (w - w.min()) / (w.mean() - w.min())
    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, h, index, threshold: float = 1.):
    _, edge_num = edge_index.size()
    edge_weights = (edge_weights + 1e-9) / (edge_weights.mean() + 1e-9) * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    # keep probability
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    edge_remove_index = np.array(list(range(h)))[sel_mask.cpu().numpy()]
    edge_remove_index_all = []
    for remove_index in edge_remove_index:
        edge_remove_index_all.extend(index[remove_index])
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove
    return edge_index


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    zero_v = torch.zeros_like(token)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token

    return data,sorted(set([i  for i in range(data.x.shape[0])])),sorted(set(data.edge_index[1][torch.where((data.edge_index[1]<node_num+data.num_hyperedges) & (data.edge_index[1]>node_num-1))[0]].cpu().numpy()))


def drop_nodes(data, aug_ratio):
    node_size = int(data.n_x[0].item())
    sub_size = int(node_size * (1 - aug_ratio))
    # hyperedge_size = int(data.num_hyperedges[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(sample_nodes, 1, edge_index, relabel_nodes=False,
                                                           flow='target_to_source')
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    sub_edge_index_orig = copy.deepcopy(sub_edge_index)
    # relabel
    node_idx = torch.zeros(2 * node_size + hyperedge_size, dtype=torch.long, device=device)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    data.x = data.x[sample_nodes]
    data.edge_index = sub_edge_index

    data.n_x = torch.tensor([sub_size])
    data.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * sub_size])
    data.norm = 0
    data.totedges = torch.tensor(sub_nodes.size(0) - sub_size)
    data.num_ori_edge = sub_edge_index.shape[1] - sub_size

    # return data, set(sub_nodes[:sub_size].cpu().numpy()),set(sub_edge_index_orig[1][torch.where((sub_edge_index_orig[1]<node_size+hyperedge_size) & (sub_edge_index_orig[1]>node_size-1))[0]].cpu().numpy())
    return data, sorted(set(sub_nodes[:sub_size].cpu().numpy())),sorted(set(sub_edge_index_orig[1][torch.where((sub_edge_index_orig[1]<node_size+hyperedge_size) & (sub_edge_index_orig[1]>node_size-1))[0]].cpu().numpy()))


def subgraph_aug(data, aug_ratio, start):
    n_walkLen = 16
    node_num, _ = data.x.size()
    he_num = data.totedges.item()
    edge_index = data.edge_index

    row, col = edge_index
    # torch.cat([row,col])
    # adj = SparseTensor(row=torch.cat([row,col]), col=torch.cat([col,row]), sparse_sizes=(node_num+he_num, he_num+node_num))
    adj = SparseTensor(row=torch.cat([row, col]), col=torch.cat([col, row]),
                       sparse_sizes=(node_num + he_num, he_num + node_num))

    node_idx = adj.random_walk(start.flatten(), n_walkLen).view(-1)
    sub_nodes = node_idx.unique()
    sub_nodes.sort()
    # sub_edge_index, _ = subgraph(sub_nodes, edge_index, relabel_nodes=True)
    # sub_edge_index, _, hyperedge_idx = subgraph(sub_nodes, edge_index, relabel_nodes=False, return_edge_mask=True)
    # data.edge_index = sub_edge_index
    # cidx = torch.where(sub_nodes >= node_num)[
    #     0].min()
    # data.x = data.x[sub_nodes[:cidx]]


    ###
    node_size = int(data.n_x[0].item())
    hyperedge_size = int(data.num_hyperedges)

    # sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(sample_nodes, 1, edge_index, relabel_nodes=False,
    #                                                        flow='target_to_source')
    sub_edge_index, _, hyperedge_idx = subgraph(sub_nodes, edge_index, relabel_nodes=False, return_edge_mask=True)


    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    sub_edge_index_orig = copy.deepcopy(sub_edge_index)
    # relabel
    node_idx = torch.zeros(2 * node_size + hyperedge_size, dtype=torch.long, device=device)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    node_keep_idx = sub_nodes[torch.where(sub_nodes<node_size)[0]]
    data.x = data.x[node_keep_idx]
    data.edge_index = sub_edge_index

    data.n_x = torch.tensor([node_keep_idx.size(0)])
    data.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * node_keep_idx.size(0)])
    data.norm = 0
    data.totedges = torch.tensor(sub_nodes.size(0) - node_keep_idx.size(0))
    data.num_ori_edge = sub_edge_index.shape[1] - node_keep_idx.size(0)

    # return data, set(sub_nodes[:sub_size].cpu().numpy()), set(sub_edge_index_orig[1][torch.where(
    #     (sub_edge_index_orig[1] < node_size + hyperedge_size) & (sub_edge_index_orig[1] > node_size - 1))[
    #     0]].cpu().numpy())

    # return data, set(node_keep_idx.cpu().numpy()),set(sub_edge_index_orig[1][torch.where((sub_edge_index_orig[1] < node_size + hyperedge_size) & (sub_edge_index_orig[1] > node_size - 1))[0]].cpu().numpy())
    return data,sorted(set(node_keep_idx.cpu().numpy().tolist())),sorted(set(sub_edge_index_orig[1][torch.where((sub_edge_index_orig[1] < node_size + hyperedge_size) & (sub_edge_index_orig[1] > node_size - 1))[0]].cpu().numpy()))

def aug(data, aug_type, start=None):
    data_aug = copy.deepcopy(data)
    if aug_type == "mask":
        data_aug, sample_nodes,sample_hyperedge = mask_nodes(data_aug, args.aug_ratio)
        return data_aug, sample_nodes,sample_hyperedge
    elif aug_type == "edge":
        data_aug, sample_nodes,sample_hyperedge = permute_edges(data_aug, args.aug_ratio, args.permute_self_edge)
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "hyperedge":
        data_aug = permute_hyperedges(data_aug, args.aug_ratio)
    elif aug_type == "subgraph":
        data_aug, sample_nodes,sample_hyperedge = subgraph_aug(data_aug, args.aug_ratio, start)
        return data_aug, sample_nodes,sample_hyperedge
    elif aug_type == "drop":
        data_aug, sample_nodes,sample_hyperedge = drop_nodes(data_aug, args.aug_ratio)
        return data_aug, sample_nodes,sample_hyperedge
    elif aug_type == "none":
        return data_aug
    elif "adapt" in aug_type:
        data_aug = adapt(data_aug, args.aug_ratio, aug_type)
    else:
        raise ValueError(f'not supported augmentation')
    return data_aug



def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))





def whole_batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []
    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]

        losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                 / (refl_sim.sum(1) + between_sim.sum(1)
                                    - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))


def batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = np.arange(0, num_nodes)
    np.random.shuffle(indices)
    i = 0
    mask = indices[i * batch_size:(i + 1) * batch_size]
    refl_sim = f(sim(z1[mask], z1))  # [B, N]
    between_sim = f(sim(z1[mask], z2))  # [B, N]
    loss = -torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                      / (refl_sim.sum(1) + between_sim.sum(1)
                         - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag()))

    return loss


def com_semi_loss(z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim[com_nodes1, com_nodes2] / (
                refl_sim.sum(1)[com_nodes1] + between_sim.sum(1)[com_nodes1] - refl_sim.diag()[com_nodes1]))


class SimCLRTau():
    def __init__(self,args,):
        super(SimCLRTau).__init__()
        self.proj1 = nn.Linear(args.hidden,args.proj)
        self.proj2 = nn.Linear(args.hidden,args.proj)
        self.tau = nn.Linear(args.proj,1)

    def forward(self,z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
        z1 = self.proj1(z1)
        z2 = self.proj2(z2)

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))

        return -torch.log(between_sim[com_nodes1, com_nodes2] / (
            refl_sim.sum(1)[com_nodes1] + between_sim.sum(1)[com_nodes1] - refl_sim.diag()[com_nodes1]))

def contrastive_loss_node(x1, x2, args, com_nodes=None):
    T = args.t
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if com_nodes is None:
        if batch_size is None:
            l1 = semi_loss(x1, x2, T)
            l2 = semi_loss(x2, x1, T)
        else:
            l1 = batched_semi_loss(x1, x2, batch_size, T)
            l2 = batched_semi_loss(x2, x1, batch_size, T)
    else:
        l1 = com_semi_loss(x1, x2, T, com_nodes[0], com_nodes[1])
        l2 = com_semi_loss(x2, x1, T, com_nodes[1], com_nodes[0])
    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret


def semi_loss_JSD(z1: torch.Tensor, z2: torch.Tensor):
    # f = lambda x: torch.exp(x / T)

    refl_sim = sim(z1, z1)
    between_sim = sim(z1, z2)
    N = refl_sim.shape[0]
    pos_score = (np.log(2) - F.softplus(- between_sim.diag())).mean()
    neg_score_1 = (F.softplus(- refl_sim) + refl_sim - np.log(2))
    neg_score_1 = torch.sum(neg_score_1) - torch.sum(neg_score_1.diag())
    neg_score_2 = torch.sum(F.softplus(- between_sim) + between_sim - np.log(2))
    neg_score = (neg_score_1 + neg_score_2) / (N * (2 * N - 1))
    return neg_score - pos_score


def contrastive_loss_node_JSD(x1, x2, args, com_nodes=None):
    T = args.t
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if com_nodes is None:
        if batch_size is None:
            l1 = semi_loss_JSD(x1, x2)
            l2 = semi_loss_JSD(x2, x1)
        else:
            l1 = batched_semi_loss(x1, x2, batch_size, T)
            l2 = batched_semi_loss(x2, x1, batch_size, T)
    else:
        l1 = com_semi_loss(x1, x2, T, com_nodes[0], com_nodes[1])
        l2 = com_semi_loss(x2, x1, T, com_nodes[1], com_nodes[0])
    ret = (l1 + l2) * 0.5
    return ret


def semi_loss_TM(z1: torch.Tensor, z2: torch.Tensor):
    # f = lambda x: torch.exp(x / T)
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    eps = 1.0
    N = z1.shape[0]
    pdist = nn.PairwiseDistance(p=2)
    pos_score = pdist(z1, z2).mean()
    neg_score_1 = torch.cdist(z1, z1, p=2)
    neg_score_2 = torch.cdist(z1, z2, p=2)
    neg_score_1 = torch.sum(neg_score_1) - torch.sum(neg_score_1.diag())
    neg_score_2 = torch.sum(neg_score_2)
    neg_score = (neg_score_1 + neg_score_2) / (N * (2 * N - 1))
    return torch.max(pos_score - neg_score + eps, 0)[0]


def contrastive_loss_node_TM(x1, x2, args, com_nodes=None):
    T = args.t
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if com_nodes is None:
        if batch_size is None:
            l1 = semi_loss_TM(x1, x2)
            l2 = semi_loss_TM(x2, x1)
        else:
            l1 = batched_semi_loss(x1, x2, batch_size, T)
            l2 = batched_semi_loss(x2, x1, batch_size, T)
    else:
        l1 = com_semi_loss(x1, x2, T, com_nodes[0], com_nodes[1])
        l2 = com_semi_loss(x2, x1, T, com_nodes[1], com_nodes[0])
    ret = (l1 + l2) * 0.5
    return ret


def sim_d(z1: torch.Tensor, z2: torch.Tensor):
    # z1 = F.normalize(z1)
    # z2 = F.normalize(z2)
    return torch.sqrt(torch.sum(torch.pow(z1 - z2, 2), 1))


def calculate_distance(z1: torch.Tensor, z2: torch.Tensor):
    num_nodes = z1.size(0)
    refl_sim = 0
    for i in range(num_nodes):
        refl_sim += (torch.sum(sim_d(z1[i:i + 1], z1)) - torch.squeeze(sim_d(z1[i:i + 1], z1[i:i + 1]))) / (
                    num_nodes - 1)
    refl_sim = refl_sim / (num_nodes)
    between_sim = torch.sum(sim_d(z1, z2)) / num_nodes
    # print(refl_sim, between_sim)


def create_hypersubgraph(data, args):
    sub_size = args.sub_size
    node_size = int(data.n_x[0].item())
    # hyperedge_size = int(data.num_hyperedges[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(sample_nodes, 1, edge_index, relabel_nodes=False,
                                                           flow='target_to_source')
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    # relabel
    node_idx = torch.zeros(2 * node_size + hyperedge_size, dtype=torch.long, device=device)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    x = data.x[sample_nodes]
    data_sub = Data(x=x, edge_index=sub_edge_index)
    data_sub.n_x = torch.tensor([sub_size])
    data_sub.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * sub_size])
    data_sub.norm = 0
    data_sub.totedges = torch.tensor(sub_nodes.size(0) - sub_size)
    data_sub.num_ori_edge = sub_edge_index.shape[1] - sub_size
    return data_sub