# -*- coding: utf-8 -*-
# @project：hot_item_mining
# @author:caojinlei
# @file: preprocess.py
# @time: 2022/05/23
from tqdm import tqdm
from simcse import SimCSE
import numpy as np
import argparse
import random
from sklearn.metrics.pairwise import cosine_similarity


def fill_zero(x, return_type=float):
    if x == "":
        if return_type == float:
            return 0.0
        else:
            return 0
    else:
        if return_type == float:
            return float(x)
        else:
            return int(x)


def data_augment(source_path, augment_source_path):
    label_count_dict = {}
    with open(source_path, 'r') as f:
        for lines in tqdm(f.readlines()):
            if lines.strip().split('\t')[0] == 'label':
                continue
            line = lines.strip().split('\t')
            label = line[0]
            if label_count_dict.get(label):
                label_count_dict[label] += 1
            else:
                label_count_dict[label] = 1
    most_label_count = max(label_count_dict.values())
    lines_list = []
    with open(source_path, 'r') as f:
        for lines in tqdm(f.readlines()):
            if lines.strip().split('\t')[0] == 'label':
                columns = lines.strip().split('\t')
                continue
            label, match_id, item_id, title, item_price_min_ch, item_price_max_ch, rating \
                , rating_count, review_count, sold_count, embed_str = lines.strip().split('\t')
            label_count = label_count_dict[label]
            augment = round(most_label_count / label_count)
            for i in range(augment):
                item_id_trans = f'f{i}' + item_id
                embed_val = np.array([float(i) for i in embed_str.split(',')])
                noise = np.random.uniform(-0.05, 0.05, 128)  # 增加噪声
                embed_val_noise = embed_val + noise
                embed_str_trans = ','.join([str(x) for x in embed_val_noise])
                lines_list.append([label, match_id, item_id_trans, title, item_price_min_ch, item_price_max_ch, rating,
                                   rating_count, review_count, sold_count, embed_str_trans])
    with open(augment_source_path, 'w') as f:
        f.writelines('\t'.join(columns) + '\n')
        for lines in lines_list:
            f.writelines('\t'.join(lines) + '\n')


def data_process(source_path, node_path, edge_path, embedding_path, threshold=0.75, top_k=30):
    node_list_info = []
    title_list = []
    emb_list = []
    node_id = 0
    columns = ['node_id', 'item_id', 'label', 'item_price_min_ch', 'item_price_max_ch', 'rating', 'rating_count',
               'review_count', 'sold_count']
    alphabet = 'abcdefghijklmnopqrstuvwxyz'  # 避免同标题
    with open(source_path, 'r') as f:
        for lines in tqdm(f.readlines()):
            try:
                if lines.strip().split('\t')[0] == 'label':
                    continue
                label, match_id, item_id, title, item_price_min_ch, item_price_max_ch, rating \
                    , rating_count, review_count, sold_count, embed_str = lines.strip().split('\t')
                node_list_info.append(
                    [node_id, item_id, fill_zero(label, int), fill_zero(item_price_min_ch),
                     fill_zero(item_price_max_ch), fill_zero(rating), fill_zero(rating_count),
                     fill_zero(review_count), fill_zero(sold_count)]
                )
                if title in title_list:
                    title = title + ' F' + ''.join(random.sample(alphabet, 5))  # 随机增加词汇
                title_list.append(title)

                emb_list.append([float(i) for i in embed_str.split(',')])
                node_id += 1

            except Exception as e:
                print(lines)
    assert len(title_list) == len(node_list_info)
    np.save(embedding_path, np.array(emb_list))  # 图片特征保存

    # 边的判定
    model = SimCSE('princeton-nlp/sup-simcse-bert-base-uncased')
    model.build_index(title_list, True, device='cuda:1')
    results = model.search(title_list, device='cuda:1', threshold=threshold, top_k=top_k)
    node_length = len(title_list)
    edge_list = []
    for i in tqdm(range(node_length)):
        src_info = node_list_info[i]
        for sentence, score in results[i]:
            index = title_list.index(sentence)
            if index == i:  # 跳过本源
                continue
            else:
                dst_info = node_list_info[index]
                # todo:还可以增加很多边的判断
                # sold_diff = abs(dst_info[-1] - src_info[-1]) / max(src_info[-1], 0.001)
                edge_list.append([src_info[0], dst_info[0], score])
    with open(edge_path, 'w') as f:
        f.writelines('src_node,dst_node,score\n')
        for edge in edge_list:
            f.writelines(','.join([str(i) for i in edge]) + '\n')

    with open(node_path, 'w') as f:
        f.writelines(','.join(columns) + '\n')
        for node in node_list_info:
            f.writelines(','.join([str(i) for i in node]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ae features were used to predict products')
    parser.add_argument("--imp", type=int, default=1000,
                        help="number of imp number")
    parser.add_argument("--top_k", type=int, default=30,
                        help="title similary top k")
    parser.add_argument('-thre', '--threshold', type=float, default=0.85,
                        help="similary threshold")
    parser.add_argument('--augment', action='store_true', default=False,
                        help='enables CUDA training(default: False)')
    args = parser.parse_args()
    imp_count = args.imp
    if_augment = args.augment
    source_path = f'../inputs/imp_{imp_count}/base/source.csv'
    node_path = f'../inputs/imp_{imp_count}/base/node.csv'
    edge_path = f'../inputs/imp_{imp_count}/base/edge.csv'
    embedding_path = f'../inputs/imp_{imp_count}/base/fea.npy'
    # augment path
    augment_source_path = f'../inputs/imp_{imp_count}/augment/source.csv'
    augment_node_path = f'../inputs/imp_{imp_count}/augment/node.csv'
    augment_edge_path = f'../inputs/imp_{imp_count}/augment/edge.csv'
    augment_embedding_path = f'../inputs/imp_{imp_count}/augment/fea.npy'
    threshold = args.threshold
    top_k = args.top_k
    if if_augment:
        data_augment(source_path, augment_source_path)
        data_process(augment_source_path, augment_node_path, augment_edge_path, augment_embedding_path,
                     threshold=threshold, top_k=top_k)
    else:
        data_process(source_path, node_path, edge_path, embedding_path, threshold=threshold, top_k=top_k)
