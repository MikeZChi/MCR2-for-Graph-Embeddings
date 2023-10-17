## Modified from DGCL code  https://haoyang.li/
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import random
import argparse
    # https://vra.github.io/2017/12/02/argparse-usage/
    # https://docs.python.org/3/library/argparse.html#module-argparse

import os
import os.path as osp
import shutil
import numpy as np
import pickle
import yaml
from copy import deepcopy
from itertools import repeat
from torch_geometric.data import DataLoader, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool

# packages for hyperparameter tuning (GridSearchCV) and cross-validation (StratifiedKFold)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


def arg_parse(DS=None): 
    # create a parser
    parser = argparse.ArgumentParser(description='DGCL Arguments.')
    parser.add_argument('--DS', type=str, default=DS)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_gc_layers', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--aug_num', type=int)
    parser.add_argument('--drop_ratio', type=float)
    parser.add_argument('--num_latent_factors', type=int) 
    parser.add_argument('--head_layers', type=int)
    parser.add_argument('--JK', type=str, choices=['last', 'sum'])
    parser.add_argument('--residual', type=int, choices=[0, 1])
    parser.add_argument('--proj', type=int, choices=[0, 1])
    parser.add_argument('--pool', type=str, choices=['mean', 'sum', 'max'])
    parser.add_argument('--fe', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--gam1', type=float, default=1.0,
                        help='gamma1 for tuning empirical loss (default: 1.0)')
    parser.add_argument('--gam2', type=float, default=10,
                        help='gamma2 for tuning empirical loss (default: 10)')
    parser.add_argument('--eps', type=float, default=2,
                        help='eps squared (default: 2)')

    args, unknown = parser.parse_known_args()
    
    # with open(f'config/{args.DS}_J.yml', 'r') as f:
    with open(f'{args.DS}_J.yml', 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in vars(args).items():
        if v is not None:
            config_yaml[k] = v
    config_ns = argparse.Namespace(**config_yaml)
    return config_ns


def setup_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def drop_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * 0.2) 
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num)) 
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = torch.nonzero(adj).t()
    data.edge_index = edge_index
    return data

def permute_edges(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * 0.2)
    edge_index = data.edge_index.transpose(0, 1).numpy()
    
    idx_add = np.random.choice(node_num, (permute_num, 2))

    edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)]
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data

def subgraph(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1 - 0.2)) 

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = torch.nonzero(adj).t()

    data.edge_index = edge_index
    return data

# graph-augmentation - mask nodes
def mask_nodes(data):
    # feat_dim： feature dimensions
    node_num, feat_dim = data.x.size()
    
    mask_num = int(node_num * 0.2)

    idx_mask = np.random.choice(node_num, mask_num, replace=False) # sample mask_num nodes
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                                    dtype=torch.float32)
    return data


def svc_classify(x, y, search): # SVM classifier
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            # search for best parameter C
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        # train the SVM classifier
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist() #select index from training set
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            # search for best parameter C
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        # train the SVM classifier
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)


def evaluate_embedding (embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    acc_val, acc = svc_classify(x, y, search)
    return acc, acc_val


class DGCL(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layer, device, args):
        super(DGCL, self).__init__()
        self.args = args
        self.device = device
        self.K = args.num_latent_factors
        self.embedding_dim = hidden_dim
        self.d = self.embedding_dim // self.K


        self.center_v = torch.rand((self.K, self.d), requires_grad=True).to(device)
        self.encoder = DisenEncoder(
            num_features=num_features,
            emb_dim=hidden_dim,
            num_layer=num_layer,
            K=args.num_latent_factors,
            head_layers=args.head_layers, # numbers of layers in MLP
            device=device,
            args=args,
            if_proj_head=args.proj > 0,
            drop_ratio=args.drop_ratio,
            graph_pooling=args.pool,
            JK=args.JK,
            residual=args.residual > 0
        )

        self.init_emb()
    
    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        z_graph, _ = self.encoder(x, edge_index, batch)
        return z_graph

    def loss_cal(self, x, x_aug):
        T = self.T
        T_c = 0.2
        B, H, d = x.size()
        ck = F.normalize(self.center_v)
        p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(x, dim=-1), ck)
        p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)
        x_abs = x.norm(dim=-1)
        x_aug_abs = x_aug.norm(dim=-1)
        x = torch.reshape(x, (B * H, d))
        x_aug = torch.reshape(x_aug, (B * H, d))
        x_abs = torch.squeeze(torch.reshape(x_abs, (B * H, 1)), 1)
        x_aug_abs = torch.squeeze(torch.reshape(x_aug_abs, (B * H, 1)), 1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (1e-8 + torch.einsum('i,j->ij', x_abs, x_aug_abs))
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(B * H), range(B * H)]
        score = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)
        p_y_xk = score.view(B, H)
        q_k = torch.einsum('bk,bk->bk', p_k_x, p_y_xk)
        q_k = F.normalize(q_k, dim=-1)
        elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
        loss = - elbo.view(-1).mean()
        return loss


class DisenEncoder(torch.nn.Module):
    def __init__(self, num_features, emb_dim, num_layer, K, head_layers, if_proj_head=False, drop_ratio=0.0,
                 graph_pooling='add', JK='last', residual=False, device=None, args=None):
        super(DisenEncoder, self).__init__()
        self.args = args
        self.device = device
        self.num_features = num_features
        self.K = K
        self.d = emb_dim // self.K
        self.num_layer = num_layer # total number of layers
        self.head_layers = head_layers
        self.gc_layers = self.num_layer - self.head_layers
        self.if_proj_head = if_proj_head
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        # All pooling methods are imported from torch_geometric.nn
        # the node representations are added together element-wise to obtain a single graph-level representation
        if self.graph_pooling == "sum" or self.graph_pooling == 'add':
            self.pool = global_add_pool
        # calculates the average of the node representations    
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        # takes the maximum value of the node representations
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        self.JK = JK
        if JK == 'last':
            pass # no action is taken
        elif JK == 'sum':
            self.JK_proj = Linear(self.gc_layers * emb_dim, emb_dim)
        else:
            assert False #invalid input for JK
        self.residual = residual
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.disen_convs = torch.nn.ModuleList()
        self.disen_bns = torch.nn.ModuleList()
        
        for i in range(self.gc_layers):
            if i == 0: # input are node features
                nn = Sequential(Linear(num_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim) # Batch normalization is a technique used in deep neural networks to improve the stability and performance of the model during training by normalizing the input data at each mini-batch.
            self.convs.append(conv)
            self.bns.append(bn)

        for i in range(self.K):
            for j in range(self.head_layers):
                if j == 0:
                    nn = Sequential(Linear(emb_dim, self.d), ReLU(), Linear(self.d, self.d))
                else:
                    nn = Sequential(Linear(self.d, self.d), ReLU(), Linear(self.d, self.d))
                conv = GINConv(nn) # still utilize the graph info
                bn = torch.nn.BatchNorm1d(self.d)

                self.disen_convs.append(conv)
                self.disen_bns.append(bn)

        self.proj_heads = torch.nn.ModuleList() # pure MLP
        for i in range(self.K):
            nn = Sequential(Linear(self.d, self.d), ReLU(inplace=True), Linear(self.d, self.d))
            self.proj_heads.append(nn)
    
    def _normal_conv(self, x, edge_index, batch):
        xs = []
        for i in range(self.gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i == self.gc_layers - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            if self.residual and i > 0:
                x += xs[i - 1]
            xs.append(x)
        if self.JK == 'last':
            return xs[-1]
        elif self.JK == 'sum':
            return self.JK_proj(torch.cat(xs, dim=-1))

    def _disen_conv(self, x, edge_index, batch):
        x_proj_list = []
        x_proj_pool_list = []
        for i in range(self.K):
            x_proj = x
            for j in range(self.head_layers):
                tmp_index = i * self.head_layers + j
                x_proj = self.disen_convs[tmp_index](x_proj, edge_index)
                x_proj = self.disen_bns[tmp_index](x_proj)
                if j != self.head_layers - 1:
                    x_proj = F.relu(x_proj)
            x_proj_list.append(x_proj)
            x_proj_pool_list.append(self.pool(x_proj, batch))
        if self.if_proj_head:
            x_proj_pool_list = self._proj_head(x_proj_pool_list)
        x_graph_multi = torch.stack(x_proj_pool_list)
        x_node_multi = torch.stack(x_proj_list)
        x_graph_multi = x_graph_multi.permute(1, 0, 2).contiguous()
        x_node_multi = x_node_multi.permute(1, 0, 2).contiguous()
        return x_graph_multi, x_node_multi

    def _proj_head(self, x_proj_pool_list):
        ret = []
        for k in range(self.K):
            # apply MLP to k latent factors
            x_graph_proj = self.proj_heads[k](x_proj_pool_list[k])
            ret.append(x_graph_proj)
        return ret

    def forward(self, x, edge_index, batch):
        if x is None: # if no node feature, use a vector of 1s
            x = torch.ones((batch.shape[0], 1)).to(device)
        h_node = self._normal_conv(x, edge_index, batch)
        h_graph_multi, h_node_multi = self._disen_conv(h_node, edge_index, batch)
        return h_graph_multi, h_node_multi

    def get_embeddings(self, loader):
        device = self.device
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)
                B, K, d = x.size()
                # The computed graph-level embeddings x are reshaped to have dimensions (B, K * d)
                x = x.view(B, K * d)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


# This class is revised from  https://haoyang.li/  DGCL.ipynb code
class TUDataset(InMemoryDataset): # load dataset
    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, aug=None, args=None):
        self.name = name
        self.cleaned = cleaned

        super(TUDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109'):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            if args.fe == 0:
                self.data.x = torch.ones((num_node, 1))
            elif args.fe == 1:
                deg_file = os.path.join(root, name, 'degree.pickle')
                if not os.path.exists(deg_file):
                    edge_nums = []
                    for i, _ in enumerate(self.slices['edge_index']):
                        if i:
                            edge_nums.append(self.slices['edge_index'][i] - self.slices['edge_index'][i - 1])
                    edge_indexs = torch.split(self.data.edge_index[1], edge_nums)
                    degs_2d = [degree(edge_index, nlist[i], dtype=torch.long) for i, edge_index in
                               enumerate(edge_indexs)]
                    degs_1d = []
                    for i in degs_2d:
                        for j in i:
                            degs_1d.append(j)

                    max_deg = int(max(degs_1d))
                    degs_1d = torch.stack(degs_1d)
                    deg_f = F.one_hot(degs_1d, num_classes=max_deg + 1).to(torch.float)
                    if not os.path.exists(deg_file):
                        with open(deg_file, 'wb') as handle:
                            pickle.dump(deg_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(deg_file, 'rb') as handle:
                        deg_f = pickle.load(handle)
                self.data.x = deg_f

            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)
        self.aug = aug

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, _= read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            try:
                item, slices = self.data[key], self.slices[key]
            except:
                continue

            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            try:
                item, slices = self.data[key], self.slices[key]
            except:
                continue
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        node_num = data.edge_index.max() + 1    # Junbin added " + 1 "
        sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)    # add self-loop

        data_group = []
        data_group.append(data)
        for i in range(self.aug - 1):
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = mask_nodes(deepcopy(data))
            elif n == 3:
                data_aug = subgraph(deepcopy(data)) # Subgraph is not used for simplicity.
            else:
                print('sample error')
                assert False

            # Some nodes may be removed in the augmented graphs, so we shall re-index nodes
            edge_idx = data_aug.edge_index.numpy()
            _, edge_num = edge_idx.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
            node_num_aug = len(idx_not_missing)
            data_aug.x = data_aug.x[idx_not_missing]
            idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                        not edge_idx[0, n] == edge_idx[1, n]]

            if self.name == 'NCI1' or self.name == 'PTC_MR' or self.name == 'REDDIT-BINARY' or self.name == 'REDDIT-MULTI-5K':
                edge_idx_tensor = torch.tensor(edge_idx)
                edge_idx_tensor = edge_idx_tensor.reshape(edge_idx_tensor.shape[0],2) # len(edge_idx[0])
                edge_idx_tensor = edge_idx_tensor.type(torch.int64)
                data_aug.edge_index = edge_idx_tensor.transpose_(0, 1)
                # data_group = data_group + (data_aug,)
                data_group.append(data_aug)
            else:
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
                # data_group = data_group + (data_aug,)
                data_group.append(data_aug)

        return data_group

# This is from MCR2  https://github.com/ryanchankh/mcr2
def one_hot(labels_int, n_classes):
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

def label_to_membership(targets, num_classes=None):
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))

    np.random.seed(32)
    # known membership; add some noise to Pi
    for j in range(len(targets)):
        random_noise = np.random.normal(0,0.05)
        random_noise = abs(random_noise)
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1 - random_noise
        
        remaining_indices = [idx for idx in range(num_classes) if idx != k]
        
        for idx in remaining_indices:
            remaining_val = random_noise/len(remaining_indices)
            Pi[idx, j, j] = remaining_val
            
    return Pi

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01, hidden_size = 0):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.hidden_size = hidden_size
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bn1 = nn.BatchNorm1d(self.hidden_size, device = device)
    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p, m = W.shape
        I = torch.eye(p)  # .cuda()
        I = I.to(device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """Empirical Compressive Loss. """
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p)
        I = I.to(device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            if torch.isnan(log_det):
                log_det = torch.logdet(I)
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p, m = W.shape
        I = torch.eye(p)  # .cuda()
        I = I.to(device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p)  # .cuda()
        I = I.to(device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if num_classes is None:
            num_classes = Y.max() + 1
        X = self.bn1(X)
        W = X.T
        Y_cpu = Y.cpu()
        Y_np = Y_cpu.numpy()
        
        Pi = label_to_membership(Y_np, num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32)
        # Put everything back in GPU
        Y = Y.to(device)
        Pi = Pi.to(device)

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()])


if __name__ == '__main__': 
    datasets_names = ['MUTAG']
    for i in range(len(datasets_names)):
        datasetname = datasets_names[i] 
        args = arg_parse(datasetname) 
        # Sets up random seed for reproducibility using the setup_seed function.
        setup_seed(args.seed)
        # Initializes dictionaries to store accuracy results during training.
        accuracies = {'result': [], 'result_val': []}
        # Sets various hyperparameters
        epochs = args.epoch
        log_interval = args.log_interval
        batch_size = args.batch
        lr = args.lr
        DS = args.DS
        # Loads the graph dataset using TUDataset from the specified path with the given dataset name (DS), and shuffles the dataset.
        path = './data'
        dataset = TUDataset(path, name=DS, args = args, aug=args.aug_num).shuffle()
        dataset_eval = TUDataset(path, name=DS, args = args, aug=1).shuffle()
        # Creates data loaders using DataLoader from PyTorch to load data in batches for training and evaluation.
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers)
        dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=args.num_workers)
        print(dataset.get_num_feature())
        try:
            dataset_num_features = dataset.get_num_feature()
        except:
            dataset_num_features = 1
        # Sets the device (CPU or GPU) based on availability of GPU.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.eps = 0.01
        args.num_latent_factors = batch_size
        args.hidden_dim = int(args.hidden_dim // args.num_latent_factors) * args.num_latent_factors
        # Defines the DGCL model using the DGCL class
        model = DGCL(
            num_features=dataset_num_features,
            hidden_dim=args.hidden_dim,
            num_layer=args.num_gc_layers,
            device=device,
            args=args
        ).to(device)
        # Defines the optimizer as Adam with the specified learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # criterion as MCR^2 with the specified hyperparameters gam1, gam2, and eps.
        criterion = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps, hidden_size = args.hidden_dim)
        # Starts the training loop for the specified number of epochs.
        for epoch in range(1, epochs + 1):
            loss_all = 0
            # Sets the model to train mode using model.train().
            model.train()

            for data in dataloader:
                #data, data_aug = data
                embeddings = []
                local_batch_size = data[0].y.shape[0]
                optimizer.zero_grad()
                for i in range(len(data)):
                    node_num, _ = data[i].x.size()
                    data[i] = data[i].to(device)
                    transferred_feats = model(data[i].x, data[i].edge_index, data[i].batch, data[i].num_graphs)
                    embeddings.append(transferred_feats.view(local_batch_size, args.num_latent_factors * transferred_feats.shape[-1]))
                    if i == 0:
                        labels = data[i].y
                    else:
                        labels = torch.cat((labels,data[i].y))
                labels_tensor = labels.float()
                embeddings = torch.stack(embeddings, dim=1)
                embeddings = embeddings.reshape(-1, embeddings.shape[-1])
                batch_idx = []
                for ii in range(local_batch_size):
                    batch_idx = batch_idx + [ii] * args.aug_num
                batch_idx = torch.from_numpy(np.array(batch_idx)).to(torch.long).to(device)
                loss, loss_empi, loss_theo = criterion(embeddings, batch_idx)
                loss_all += loss.item()
                loss.backward()
                optimizer.step()
            # Print the training loss
            # if epoch % 5 < 0.1:
                # print('loss %.4f' % (loss_all / len(dataloader)))
            # log_interval is defaulted to be 5
            if log_interval > 0 and epoch % log_interval == 0:
                # sets the model to evaluation mode using model.eval()
                model.eval()
                # gets the embeddings and ground truth labels using model.encoder.get_embeddings
                emb, y = model.encoder.get_embeddings(dataloader_eval)
                # evaluates the embeddings using evaluate_embedding function
                result, result_val = evaluate_embedding(emb, y)
                accuracies['result'].append(result)
                accuracies['result_val'].append(result_val)
        
        results = 0
        results_val = 0
        results_mean = np.array(accuracies['result']).mean()
        results_mean = round(results_mean,4)
        results_val_mean = np.array(accuracies['result_val']).mean()
        results_val_mean = round(results_val_mean,4)
        results_std = np.array(accuracies['result']).std()
        results_std = round(results_std,4)
        
        results_val_std = np.array(accuracies['result_val']).std()
        results_val_std = round(results_val_std,4)
        accuracies['result'] = []
        accuracies['result_val'] = []
        
        print(datasetname,results_mean,results_std,results_val_mean,results_val_std)
    
