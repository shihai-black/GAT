import argparse
import time
import datetime
import os
import dgl
import torch
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, classification_report,roc_auc_score

from inputs.datasets import MyDataSet
from models.GAT import GAT
from utils.Logginger import init_logger
from utils.config import PATH, AUGMENT_PATH, imp_count
from utils.utils import focal_loss
from callback.modelcheckpoint import ModelCheckPoint
from callback.tensorboard_pytorch import net_board, loss_board

logger = init_logger('hot_mining', './output/')
today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def arguments():
    parser = argparse.ArgumentParser(description='Ae features were used to predict products')
    parser.add_argument('--model', type=str, default='GAT', metavar='N',
                        help='Which model to choose(default: GAT)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--num_heads', type=int, default=4, metavar='N',
                        help='number of hidden attention heads (default: 4)')
    parser.add_argument('--num_out_heads', type=int, default=1, metavar='N',
                        help='number of output attention heads (default: 1)')
    parser.add_argument('--num_layers', type=int, default=3, metavar='N',
                        help='number of hidden layers (default: 3)')
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training(default: False)')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='whether or not use augment data')
    parser.add_argument('--seed', type=int, default=1111, metavar='S',
                        help='random seed (default: 1111)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Whether or not to save(default: y)')
    return parser.parse_args()


def train(args, model, train_dataloader, loss_func, optimizer, logger, epoch):
    model.train()
    count = 0
    count_loss = 0
    start_time = time.time()
    for step, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        y_hat = model(blocks, x)
        # _, indices = torch.max(y_hat, dim=1)
        # loss = loss_func(_, y.float())
        loss = loss_func(y_hat, y)
        count_loss += loss.item()
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info('Train Epoch: {} | Process: [{}/{} ({:.2f}%)] | Loss: {:.4f} | Time(s): {:.2f}\t'.format(
            epoch, step * args.batch_size + len(y), len(train_dataloader.indices),
                   100 * (step * args.batch_size + len(y)) / len(train_dataloader.indices),
            loss.item(), time.time() - start_time))
    logger.info('====> Epoch: {} | Average loss: {:.4f} | Cost Time(s): {:.2f}\t'.
                format(epoch, count_loss / count, time.time() - start_time))
    result_loss = count_loss / count
    return result_loss


@torch.no_grad()
def evaluate(model, eval_dataloader, loss_func, logger, mode):
    if mode == 'test':
        model.eval()
    y_list = []
    y_hat_list = []
    count_loss = 0
    count = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(eval_dataloader):
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        y_hat = model(blocks, x)
        _, indices = torch.max(y_hat, dim=1)
        # loss = loss_func(_, y.float())
        loss = loss_func(y_hat, y)
        count_loss += loss.item()
        count += 1
        y_list.append(y)
        y_hat_list.append(indices)
    y_list_trans = torch.cat(y_list).cpu().numpy().tolist()
    y_hat_list_trans = torch.cat(y_hat_list).cpu().numpy().tolist()
    f1 = f1_score(y_list_trans, y_hat_list_trans, average='weighted')
    target_names = ["动销", "非动销"]
    cf_matrix = confusion_matrix(y_list_trans, y_hat_list_trans)
    cr = classification_report(y_list_trans, y_hat_list_trans, target_names=target_names)
    auc = roc_auc_score(y_list_trans, y_hat_list_trans)
    logger.info('====> Mode: {} | Average loss: {:.4f} | F1_Score:{}'.format(mode, count_loss / count, f1))
    logger.info(f'====> Confusion Matrix\t {cf_matrix}')
    logger.info(f'====> Classification Report\t {cr}')
    logger.info(f'====> AUC score \t {auc}')
    result_loss = count_loss / count
    return result_loss


def cmd_entry(args, logger):
    # 载入数据
    logger.info('Load graph .....')
    pwd = os.getcwd()
    if args.augment:
        path = AUGMENT_PATH
    else:
        path = PATH
    node_path = os.path.join(pwd, path['node_path'])
    edge_path = os.path.join(pwd, path['edge_path'])
    embedding_path = os.path.join(pwd, path['emb_path'])
    data = MyDataSet(node_path, edge_path, embedding_path, ',')
    g = data[0]

    # 全局信息
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    logger.info("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
                (n_edges, n_classes,
                 train_mask.int().sum().item(),
                 val_mask.int().sum().item(),
                 test_mask.int().sum().item()))

    # 是否使用gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:1" if args.cuda else "cpu")

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # 构造模型
    logger.info('Loading network started ...')
    heads = ([args.num_heads] * (args.num_layers - 1)) + [args.num_out_heads]
    model = GAT(g, args.num_layers, num_feats, args.num_hidden, n_classes, heads,
                F.elu, args.in_drop, args.attn_drop, args.negative_slope, args.residual).to(device)
    if args.augment:
        model_name = f'{args.model}_{imp_count}_augment'
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        model_name = f'{args.model}_{imp_count}'
        # loss_func = focal_loss(0.5, 2, 2)
        loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 数据构造
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 5, 2], prefetch_node_feats=['feat'],
                                                        prefetch_labels=['label'])
    train_ids = data.train_ids
    test_ids = data.test_ids
    val_ids = data.val_ids
    train_dataloader = dgl.dataloading.DataLoader(
        g, train_ids, sampler, device=device, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataloader = dgl.dataloading.DataLoader(
        g, test_ids, sampler, device=device, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_dataloader = dgl.dataloading.DataLoader(
        g, val_ids, sampler, device=device, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_statistics = Counter(labels[train_mask].tolist())
    test_statistics = Counter(labels[test_mask].tolist())
    valid_statistics = Counter(labels[val_mask].tolist())
    logger.info(f'Label Statistics\t '
                f'Train : {train_statistics}\t Test:{test_statistics}\t valid:{valid_statistics}')
    if args.save:
        save_module = ModelCheckPoint(model=model, optimizer=optimizer, log=logger, filename='output/save_dict/',
                                      name=model_name)
        state = save_module.save_info(epoch=0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, train_dataloader, loss_func, optimizer, logger, epoch)
        valid_loss = evaluate(model, valid_dataloader, loss_func, logger, 'valid')
        loss_board(f'./output/logs/{model_name}/', 'train', 'loss', train_loss, valid_loss, epoch)
        if args.save:
            state, early_stop = save_module.step_save(state, valid_loss)
            if early_stop:
                break
    test_loss = evaluate(model, test_dataloader, loss_func, logger, 'test')


if __name__ == '__main__':
    args = arguments()
    torch.manual_seed(args.seed)
    print(args)
    cmd_entry(args, logger)
