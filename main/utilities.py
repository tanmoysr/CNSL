
import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from sklearn.metrics import precision_recall_curve

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def adj_process(adj):
    """build symmetric adjacency matrix"""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj
def sparse_tensor_converter(adj):
    idx, idy, val = sp.find(adj)
    indices = np.vstack((idx, idy))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(val)
    shape = adj.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sampling(inverse_pairs): # Why sampling only 10%?
    diffusion_count = []
    for i, pair in enumerate(inverse_pairs):
        diffusion_count.append(pair[:, 1].sum())
    diffusion_count = torch.Tensor(diffusion_count)
    top_k = diffusion_count.topk(int(0.1*inverse_pairs.shape[0])).indices
    return top_k


def diffusion_evaluation(adj_matrix, seed, diffusion='LT'):
    total_infect = 0
    G = nx.from_scipy_sparse_matrix(adj_matrix)

    for i in range(10):

        if diffusion == 'LT':
            model = ep.ThresholdModel(G)
            config = mc.Configuration()
            for n in G.nodes():
                config.add_node_configuration("threshold", n, 0.5)
        elif diffusion == 'IC':
            model = ep.IndependentCascadesModel(G)
            config = mc.Configuration()
            for e in G.edges():
                config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])
        elif diffusion == 'SIS':
            model = ep.SISModel(G)
            config = mc.Configuration()
            config.add_model_parameter('beta', 0.001)
            config.add_model_parameter('lambda', 0.001)
        else:
            raise ValueError('Only IC, LT and SIS are supported.')

        config.add_model_initial_configuration("Infected", seed)

        model.set_initial_status(config)

        iterations = model.iteration_bunch(100)

        node_status = iterations[0]['status']

        seed_vec = np.array(list(node_status.values()))

        for j in range(1, len(iterations)):
            node_status.update(iterations[j]['status'])

        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1

        total_infect += inf_vec.sum()
    total_nodes = G.number_of_nodes()
    infected_nodes = total_infect / 10
    percentage_infected = (infected_nodes/total_nodes)*100
    return [percentage_infected, total_infect / 10]# infected_nodes

class MultipleOptimizer(object):
    '''
    opt = MultipleOptimizer(optimizer1(params1, lr=lr1),
                        optimizer2(params2, lr=lr2))

    loss.backward()
    opt.zero_grad()
    opt.step()
    https://discuss.pytorch.org/t/two-optimizers-for-one-model/11085/7
    '''
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def find_bestThreshold(labels,preds):
    # calculate pr curve
    precision, recall, thresholds = precision_recall_curve(labels, preds)

    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    return thresholds[ix]

def precision_at_k_score(y_true, y_pred_proba, k=100, pos_label=1):
    topk = [
        y_true_ == pos_label
        for y_true_, y_pred_proba_
        in sorted(
            zip(y_true, y_pred_proba),
            key=lambda y: y[1],
            reverse=True
        )[:k]
    ]
    return sum(topk) / len(topk)


def precision_at_k(df: pd.DataFrame, k: int, y_test: str, y_pred: str) -> float:
    """
    Function to compute precision@k for an input boolean dataframe

    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output

    Output:
        Floating-point number of precision value for k items
    """
    # extract the k rows
    dfK = df.head(k)
    # compute number of recommended items @k
    denominator = dfK[y_pred].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator / denominator
    else:
        return None


def recall_at_k(df: pd.DataFrame, k: int, y_test: str, y_pred: str) -> float:
    """
    Function to compute recall@k for an input boolean dataframe

    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output

    Output:
        Floating-point number of recall value for k items
    """
    # extract the k rows
    dfK = df.head(k)
    # compute number of all relevant items
    denominator = df[y_test].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator / denominator
    else:
        return None
