#!/usr/bin/env python
# encoding: utf-8

import os
import time
import torch
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

from scripts.layers import *
from scripts.models import *
from scripts.preprocessing import *
from collections import defaultdict
from torch_sparse import SparseTensor, coalesce
from convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_geometric.utils import dropout_adj, degree, to_undirected, k_hop_subgraph, subgraph
from torch_geometric.data import Data
from scripts.parser_data import parser_data
from sklearn.metrics import f1_score
from scripts.utils import *


if __name__ == '__main__':
    start = time.time()

    data = 'cora'
    args = parser_data(data)
    fix_seed(args.seed)
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()

    # # Part 1: Load data

    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed']

    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100',
                      'house-committees-100']

    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            # p2raw = '../data/AllSet_all_raw_data/'
            p2raw = '../data/raw_data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,
                                         feature_noise=f_noise,
                                         p2raw=p2raw)
        else:
            if dname in ['cora', 'citeseer', 'pubmed']:
                p2raw = '../data/raw_data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/raw_data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/raw_data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = '../data/raw_data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, root='../data/raw_data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw=p2raw)
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        # if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
        # if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.num_hyperedges])

    # ipdb.set_trace()
    #     Preprocessing
    # if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)  # node to edge
        if args.add_self_loop:
            data = Add_Self_Loops(data) 
        if args.exclude_self:
            data = expand_edge_index(data)

        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')

    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)
    #     ipdb.set_trace()

    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ['HCHA', 'HGNN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ['UniGCNII']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        # Compute degV and degE
        if args.cuda in [0, 1, 2, 3]:
            device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        args.device = device
        (row, col), value = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)

        degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
        from torch_scatter import scatter

        degE = scatter(degV[V], E, dim=0, reduce='mean')
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        args.UniGNN_degV = degV
        args.UniGNN_degE = degE

        V, E = V.cpu(), E.cpu()
        del V
        del E

    #     Get splits
    split_idx_lst = []
    for run in range(args.runs):  # how many runs
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)  # train test split
        split_idx_lst.append(split_idx)  # the list of data splitting

    # # Part 2: Load model

    model = parse_method(args, data)
    # put things to device
    if args.cuda in [0, 1, 2, 3]:
        device = torch.device('cuda:' + str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    data = data.to(device)
    data_pre = copy.deepcopy(data)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)

    num_params = count_parameters(model)

    # # Part 3: Main. Training + Evaluation

    logger = Logger(args.runs, args)

    criterion = nn.NLLLoss()
    eval_func = eval_acc

    model.train()
    # print('MODEL:', model)

    ### Training loop ###
    he_index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        he_index[he].append(i)  # 字典  he_index[edge_index[1]] = idx
    runtime_list = []
    if args.mode == "JSD":
        contrastive_loss = contrastive_loss_node_JSD
    elif args.mode == "TM":
        contrastive_loss = contrastive_loss_node_TM
    else:
        contrastive_loss = contrastive_loss_node
    for run in tqdm(range(args.runs)):
    # for run in range(args.runs):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            # if args.p_lr:
            #     optimizer = torch.optim.Adam(model.linear.parameters(), lr=args.lr, weight_decay=args.wd)
            # else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        #     This is for HNHN only
        #     if args.method == 'HNHN':
        #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
        if args.p_lr:
            optimizer_p = torch.optim.Adam(model.parameters(), lr=args.p_lr, weight_decay=args.wd)
        best_val = float('-inf')
        # hyperedge_idx = [i for i in range(data.n_x,data.n_x+data.num_hyperedges)]
        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()
            # cl loss
            if args.m_l:
                # if data_pre.n_x[0].item() <= args.sub_size:
                if data_pre.n_x <= args.sub_size:
                    data_sub = data_pre
                else:
                    data_sub = create_hypersubgraph(data_pre, args)  ###

                data_sub1 = copy.deepcopy(data_sub)
                data_sub2 = copy.deepcopy(data_sub)
                if args.aug1 == "subgraph" or args.aug1 == "drop":
                    node_num = data_sub1.x.shape[0]
                    n_walk = 128 if node_num > 128 else 8
                    start = torch.randint(0, node_num, size=(n_walk,), dtype=torch.long).to(device)
                    data_sub1 = data_sub1.to(device)
                    data_aug1, nodes1, hyperedge1 = aug(data_sub1, args.aug1, start)

                else:
                    data_sub1 = data_sub1.to(device)
                    cidx = data_sub1.edge_index[1].min()
                    data_sub1.edge_index[1] -= cidx
                    # must starts from zero
                    data_sub1 = data_sub1.to(device)
                    # data_aug1 = aug(data_sub, args).to(device)
                    data_aug1,nodes1,hyperedge1 = aug(data_sub1, args.aug1)
                    data_aug1 = data_aug1.to(device)
                    # nodes1 = set([i for i in range(data_sub1.x.size()[0])])
                    data_aug1.edge_index[1] += cidx
                # hyperedge_idx1 = torch.tensor([i for i in range(data_aug1.x.shape[0], data_aug1.x.shape[0] + len(hyperedge1))]).to(device)
                hyperedge_idx1 = torch.tensor(list(range(data_aug1.x.shape[0], data_aug1.x.shape[0] + len(hyperedge1)))).to(device)


                def edge_embed(idx,data_aug):

                    return data_aug.edge_index[0][torch.where(data_aug.edge_index[1] == idx)[0]]




                data1_node2edge = [edge_embed(i, data_aug1) for i in hyperedge_idx1]

                data1_edgeidx_l,data1_node2edge_sample = [],[]
                for i in range(len(data1_node2edge)):
                    if torch.numel(data1_node2edge[i]) > 0:
                        data1_edgeidx_l.append(i)
                        data1_node2edge_sample.append(data1_node2edge[i])


                data1_edgeidx =  data_aug1.x.shape[0] + torch.tensor(data1_edgeidx_l).to(device)

                pgd1 = torch.rand_like(data_aug1.x)
                data_aug_pgd1 = data_aug1.clone()
                data_aug_pgd1.x = data_aug1.x + pgd1

                out1,edge1 = model.forward_global_local(data_aug_pgd1,data1_node2edge_sample,data1_edgeidx,device)

                if args.aug2 == "subgraph" or args.aug2 == "drop":
                    node_num = data_sub2.x.shape[0]
                    n_walk = 128 if node_num > 128 else 8
                    start = torch.randint(0, node_num, size=(n_walk,), dtype=torch.long).to(device)
                    data_sub2 = data_sub2.to(device)
                    data_aug2, nodes2, hyperedge2 = aug(data_sub2, args.aug2, start)
                else:
                    data_sub2 = data_sub2.to(device)
                    cidx = data_sub2.edge_index[1].min()
                    data_sub2.edge_index[1] -= cidx
                    # must starts from zero
                    data_sub2 = data_sub2.to(device)
                    data_aug2,nodes2, hyperedge2 = aug(data_sub2, args.aug2)
                    data_aug2 = data_aug2.to(device)
                    data_aug2.edge_index[1] += cidx


                hyperedge_idx2 = torch.tensor(list(range(data_aug2.x.shape[0], data_aug2.x.shape[0] + len(hyperedge2)))).to(device)


                data2_node2edge = [edge_embed(i, data_aug2) for i in hyperedge_idx2]


                data2_edgeidx_l, data2_node2edge_sample = [], []
                for i in range(len(data2_node2edge)):
                    if torch.numel(data2_node2edge[i]) > 0:
                        data2_edgeidx_l.append(i)
                        data2_node2edge_sample.append(data2_node2edge[i])

                data2_edgeidx = data_aug2.x.shape[0] + torch.tensor(data2_edgeidx_l)

                pgd2 = torch.rand_like(data_aug2.x)
                data_aug_pgd2= data_aug2.clone()
                data_aug_pgd2.x = data_aug2.x + pgd2

                out2, edge2 = model.forward_global_local(data_aug_pgd2,data2_node2edge_sample,data2_edgeidx,device)

                # if args.aug1 in ['drop','subgraph'] or args.aug2 in ['drop','subgraph']:
                com_sample = list(set(nodes1) & set(nodes2))
                dict_nodes1, dict_nodes2 = {value: i for i, value in enumerate(nodes1)}, {value: i for
                                                                                                 i, value in
                                                                                                 enumerate(
                                                                                                    nodes2)}
                com_sample1, com_sample2 = [dict_nodes1[value] for value in com_sample], [dict_nodes2[value] for
                                                                                          value in com_sample]
                loss_cl = model.get_loss(out1, out2, args.t, [com_sample1, com_sample2])

                com_edge = list(set(data1_edgeidx.tolist()) & set(data2_edgeidx.tolist()))


                dict_edge1, dict_edge2 = {value: i for i, value in enumerate(data1_edgeidx.tolist())}, {value: i for
                                                                                            i, value in
                                                                                            enumerate(
                                                                                                data2_edgeidx.tolist())}

                com_edge1, com_edge2 = [dict_edge1[value] for value in com_edge], [dict_edge2[value] for
                                                                                   value in com_edge]
                loss_cl_gl = model.get_loss(edge1, edge2, args.t, [com_edge1, com_edge2])

            else:
                loss_cl = 0
            # sup loss
            if args.linear:
                out = model.forward_finetune(data)
            else:
                out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss += args.m_l * (loss_cl+loss_cl_gl)


            loss.backward()
            optimizer.step()

            time2 = time.time()
            if args.linear:
                result = evaluate_finetune(model, data, split_idx, eval_func)
            else:
                result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:6])

        end_time = time.time()
        runtime_list.append(end_time - start_time)

        logger.print_statistics(run)
        end = time.time()
        mins = (end - start) / 60
        print("The running time is {}".format(mins))

    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val_acc, best_test_acc, test_f1 = logger.print_statistics()
    res_root = 'hyperparameter_tunning/result_jsd/'
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        if args.p_lr:
            cur_line = f'{args.method}_{args.m_l}_{args.lr}_{args.wd}_{args.sub_size}_{args.heads}_aug_{args.aug}_ratio_{str(args.aug_ratio)}_t_{str(args.t)}_plr_{str(args.p_lr)}_pepoch_{str(args.p_epochs)}_player_{str(args.p_layer)}_phidden_{str(args.p_hidden)}_drop_{str(args.dropout)}_train_{str(args.train_prop)}'
            if args.add_e:
                cur_line += "_add_e"
        else:
            cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}_{str(args.dropout)}_train_{str(args.train_prop)}'
        cur_line += f',{args.aug1,args.aug2}'
        cur_line += f',{best_val_acc.mean():.3f} ± {best_val_acc.std():.3f}'
        cur_line += f',{best_test_acc.mean():.3f} ± {best_test_acc.std():.3f}'
        cur_line += f',{test_f1.mean():.3f} ± {test_f1.std():.3f}'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s'
        cur_line += f',{avg_time // 60}min{(avg_time % 60):.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')
    end = time.time()
    mins = (end - start) / 60
    # print("The running time is {}".format(mins))
    print('All done! Exit python code')
    quit()

