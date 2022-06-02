# -*- coding: utf-8 -*-
# @project：hot_item_mining
# @author:caojinlei
# @file: config.py
# @time: 2022/05/24
imp_count = 1000
model_name = 'GAT_1000'
PATH = {
    'node_path': f'inputs/imp_{imp_count}/base/node.csv',
    'edge_path': f'inputs/imp_{imp_count}/base/edge.csv',
    'emb_path': f'inputs/imp_{imp_count}/base/fea.npy'
}
AUGMENT_PATH = {
    'node_path': f'inputs/imp_{imp_count}/augment/node.csv',
    'edge_path': f'inputs/imp_{imp_count}/augment/edge.csv',
    'emb_path': f'inputs/imp_{imp_count}/augment/fea.npy'
}

PREDICT_PATH = {
    'node_path': 'inputs/predict/node.csv',
    'edge_path': 'inputs/predict/edge.csv',
    'id_path': '/data/cyt/similar_gallery/aliexpress/resource/meta.csv',
    'source_path': 'inputs/predict/source.csv',
    'source_emb_path': '/data/cyt/similar_gallery/aliexpress/resource/meta.feat.npy',
    'emb_path': 'inputs/predict/fea.npy',
    'cate_path': 'inputs/aliexpress_cate.csv',
    'out_path': f'inputs/predict/result_{model_name}.csv'
}
MODEL_PATH = {
    'model_best_path': f'output/save_dict/{model_name}-model_best.pth'
}
