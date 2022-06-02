# -*- coding: utf-8 -*-
# @project：hot_item_mining
# @author:caojinlei
# @file: datasets.py
# @time: 2022/05/23
import torch
import pandas as pd
import dgl
import numpy as np

from collections import Counter
from dgl.data import DGLDataset
from sklearn.model_selection import train_test_split


class MyDataSet(DGLDataset):
    def __init__(self, node_info_path, edge_path, embedding_path, sep):
        self.node_info_path = node_info_path
        self.embedding_path = embedding_path
        self.edge_path = edge_path
        self.sep = sep
        super(MyDataSet, self).__init__(name='hot_item')

    def process(self):
        nodes_data = pd.read_csv(self.node_info_path, self.sep)
        edges_data = pd.read_csv(self.edge_path, self.sep)
        embed = np.load(self.embedding_path)
        node_features = torch.from_numpy(embed) # 节点特征
        node_labels = torch.from_numpy(nodes_data['label'].astype('category').cat.codes.to_numpy())
        edge_features = torch.from_numpy(edges_data['score'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src_node'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst_node'].to_numpy())
        self.num_labels = nodes_data['label'].nunique()
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features.to(torch.float32)
        self.graph.ndata['label'] = node_labels.to(torch.long)
        self.graph.edata['weight'] = edge_features

        # 设置掩盖码
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.train_ids = torch.arange(0, n_train, dtype=torch.int64)
        self.val_ids = torch.arange(n_train, n_train + n_val, dtype=torch.int64)
        self.test_ids = torch.arange(n_train + n_val, n_nodes, dtype=torch.int64)
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, item):
        return self.graph

    def __len__(self):
        return 1


if __name__ == '__main__':
    node_path = './imp_1000/base/node.csv'
    edge_path = './imp_1000/base/edge.csv'
    embedding_path = './imp_1000/base/fea.npy'
    sep = ','
    dataset = MyDataSet(node_path, edge_path, embedding_path, sep)
    graph = dataset[0]

    print(graph)
    sampler = dgl.dataloading.NeighborSampler([15, 10, 5], prefetch_node_feats=['feat'],
                                                        prefetch_labels=['label'])
    train_ids = dataset.train_ids
    test_ids = dataset.test_ids
    val_ids = dataset.val_ids
    train_dataloader = dgl.dataloading.DataLoader(
        graph, train_ids, sampler, device='cuda:1', batch_size=1024, shuffle=True, drop_last=False)
    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_ids, sampler, device='cuda:1', batch_size=1024, shuffle=True, drop_last=False)
    val_dataloader = dgl.dataloading.DataLoader(
        graph, val_ids, sampler, device='cuda:1', batch_size=1024, shuffle=True, drop_last=False)
    for input_nodes, output_nodes, blocks in train_dataloader:
        print('train')
    # for input_nodes, output_nodes, blocks in test_dataloader:
    #     print('test')
    # for input_nodes, output_nodes, blocks in val_dataloader:
    #     print('val')
