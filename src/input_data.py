from util import *
import torch
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from typing import Union, Tuple
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx, to_dense_adj, negative_sampling
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fix_seed()

"""
patent_*** = (num_patent, num_***) num_patent = 44564
"""

patent_company = load_binary('/app/data/patent_company.pkl')
patent_term = load_binary('/app/data/patent_term.pkl')
cpc = clamp(torch.matmul(patent_company.T, patent_company), 0, 1).to('cpu')
cpc = csr_matrix(cpc)
adj_orig = cpc

# prepare for training
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(cpc)
# adj_train = cpc
if os.path.exists(f'/app/data/training/adj_norm.pkl'):
    adj_norm = load_binary('/app/data/training/adj_norm.pkl')
    adj_label = load_binary('/app/data/training/adj_label.pkl')
    norm = load_binary('/app/data/training/norm.pkl')
    weight_tensor = load_binary('/app/data/training/weight_tensor.pkl')
    print('loaded adj_norm, adj_label, norm, weight_tensor')
else:
    adj_norm, adj_label, norm, weight_tensor = prepare_adj_for_training(adj_train)
    save_binary(adj_norm, '/app/data/training/adj_norm.pkl')
    save_binary(adj_label, '/app/data/training/adj_label.pkl')
    save_binary(norm, '/app/data/training/norm.pkl')
    save_binary(weight_tensor, '/app/data/training/weight_tensor.pkl')
    print('saved adj_norm, adj_label, norm, weight_tensor')


cpt = clamp(torch.matmul(patent_company.T, patent_term), 0, 1).to('cpu').T
cpt_tensor = cpt
cpt = csr_matrix(cpt)


if os.path.exists(f'/app/data/training/bi_train.pkl'):
    bi_train = load_binary('/app/data/training/bi_train.pkl')
    train_edges_bi = load_binary('/app/data/training/train_edges_bi.pkl')
    val_edges_bi = load_binary('/app/data/training/val_edges_bi.pkl')
    val_edges_false_bi = load_binary('/app/data/training/val_edges_false_bi.pkl')
    test_edges_bi = load_binary('/app/data/training/test_edges_bi.pkl')
    test_edges_false_bi = load_binary('/app/data/training/test_edges_false_bi.pkl')
    print('loaded bi_train, train_edges_bi, val_edges_bi, val_edges_false_bi, test_edges_bi, test_edges_false_bi')
else:
    bi_train, train_edges_bi, val_edges_bi, val_edges_false_bi, test_edges_bi, test_edges_false_bi = mask_test_edges_for_bipartite(cpt)
    save_binary(bi_train, '/app/data/training/bi_train.pkl')
    save_binary(train_edges_bi, '/app/data/training/train_edges_bi.pkl')
    save_binary(val_edges_bi, '/app/data/training/val_edges_bi.pkl')
    save_binary(val_edges_false_bi, '/app/data/training/val_edges_false_bi.pkl')
    save_binary(test_edges_bi, '/app/data/training/test_edges_bi.pkl')
    save_binary(test_edges_false_bi, '/app/data/training/test_edges_false_bi.pkl')
    print('saved bi_train, train_edges_bi, val_edges_bi, val_edges_false_bi, test_edges_bi, test_edges_false_bi')
# bi_train = cpt

if os.path.exists(f'/app/data/training/bi_adj_norm.pkl'):
    bi_adj_norm = load_binary('/app/data/training/bi_adj_norm.pkl')
    biadj_label = load_binary('/app/data/training/biadj_label.pkl')
    norm_bi = load_binary('/app/data/training/norm_bi.pkl')
    weight_tensor_bi = load_binary('/app/data/training/weight_tensor_bi.pkl')
    print('loaded bi_adj_norm, biadj_label, norm_bi, weight_tensor_bi')
else:
    bi_adj_norm, biadj_label, norm_bi, weight_tensor_bi = prepare_biadj_for_training(bi_train)
    save_binary(bi_adj_norm, '/app/data/training/bi_adj_norm.pkl')
    save_binary(biadj_label, '/app/data/training/biadj_label.pkl')
    save_binary(norm_bi, '/app/data/training/norm_bi.pkl')
    save_binary(weight_tensor_bi, '/app/data/training/weight_tensor_bi.pkl')
    print('saved bi_train, train_edges_bi, val_edges_bi, val_edges_false_bi, test_edges_bi, test_edges_false_bi, bi_adj_norm, biadj_label, norm_bi, weight_tensor_bi')


