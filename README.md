# README

## 简介

GAT模型用于选品

## 主要改动

1. 数据预处理模块
2. 自定义模型process模块
3. 增加batch size模块，避免图太大导致的内存溢出
4. 增加focal loss模块用于不均衡数据

## 运行

**数据预处理**

```python
python3 preprocess.py  --augment
```

**模型训练**

```python
python3 run.py --save --cuda --batch_size 10000 --augment
```


## 参考

paper link：https://arxiv.org/abs/1710.10903

reference code：https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat

blog：https://zhuanlan.zhihu.com/p/81350196