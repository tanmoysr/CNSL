# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import List
from configuration import args
## Inferece Libraries
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.bn = nn.BatchNorm1d(num_features=latent_dim)

    def forward(self, x):
        h_ = self.FC_input(x)
        # h_ = F.relu(self.FC_input(x))

        # h_ = F.relu(self.FC_input2(h_))
        # h_ = self.FC_input2(h_)

        h_ = F.relu(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim, seed_pred = False, seed_plus = False):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.seed_pred = seed_pred
        self.seed_plus = seed_plus

        # self.prelu = nn.PReLU()

    def forward(self, x):
        h = F.relu(self.FC_input(x))
        # h = self.FC_input(x)
        # h = F.relu(self.FC_hidden_1(h))
        # h = F.relu(self.FC_hidden_2(h))
        # x_hat = torch.sigmoid(self.FC_output(h))
        h = self.FC_hidden_1(h)
        h = self.FC_hidden_2(h)
        if self.seed_pred:
            if self.seed_plus:
                x_hat_pre = self.FC_output(h)
                x_hat = torch.cat((torch.sigmoid(x_hat_pre[:, 0]).unsqueeze(1),
                                   F.relu(x_hat_pre[:, 1:])), 1)
            else:
                x_hat = torch.sigmoid(self.FC_output(h))
        else:
            x_hat = F.relu(self.FC_output(h))
        return x_hat


class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        std = torch.exp(0.5 * var)  # standard deviation
        epsilon = torch.randn_like(var)
        return mean + std * epsilon

    def forward(self, x, adj=None):
        if adj != None:
            mean, log_var = self.Encoder(x, adj)
        else:
            mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, log_var)  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


class VAEModelComb(nn.Module):
    def __init__(self, Encoder_sd, Encoder_ft, Encoder_sd_ft, Decoder_sd, Decoder_ft, Decoder_sd_ft):
        super(VAEModelComb, self).__init__()
        self.Encoder_sd = Encoder_sd
        self.Encoder_ft = Encoder_ft
        self.Encoder_sd_ft = Encoder_sd_ft

        self.Decoder_sd = Decoder_sd
        self.Decoder_ft = Decoder_ft
        self.Decoder_sd_ft = Decoder_sd_ft

    def reparameterization(self, mean, var):
        std = torch.exp(0.5 * var)  # standard deviation
        epsilon = torch.randn_like(var)
        return mean + std * epsilon

    def forward(self, sd, ft, sd_ft):
        mean_sd, log_var_sd = self.Encoder_sd(sd)
        mean_ft, log_var_ft = self.Encoder_ft(ft)
        mean_sd_ft, log_var_sd_ft = self.Encoder_sd_ft(sd_ft)

        z_sd = self.reparameterization(mean_sd, log_var_sd)  # takes exponential function (log var -> var)
        z_ft = self.reparameterization(mean_ft, log_var_ft)
        z_sd_ft = self.reparameterization(mean_sd_ft, log_var_sd_ft)

        z_sd_plus = torch.cat((z_sd, z_sd_ft), -1)
        z_ft_plus = torch.cat((z_ft, z_sd_ft), -1)

        x_hat_sd = self.Decoder_sd(z_sd_plus)
        x_hat_ft = self.Decoder_ft(z_ft_plus)
        x_hat_sd_ft = self.Decoder_sd_ft(z_sd_ft)

        x_hat = [x_hat_sd, x_hat_ft, x_hat_sd_ft]
        mean = [mean_sd, mean_ft, mean_sd_ft]
        log_var = [log_var_sd, log_var_ft, log_var_sd_ft]

        return x_hat, mean, log_var


class GNNModel(nn.Module):
    def __init__(self, input_dim, hiddenunits: List[int], num_classes, prob_matrix, bias=True, drop_prob=0.5):
        super(GNNModel, self).__init__()

        self.input_dim = input_dim

        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.prob_matrix = nn.Parameter((torch.FloatTensor(prob_matrix)), requires_grad=False)

        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)

        self.act_fn = nn.ReLU()

    def forward(self, seed_vec):

        for i in range(self.input_dim - 1):
            if i == 0:
                mat = self.prob_matrix.T @ seed_vec.T
                attr_mat = torch.cat((seed_vec.T.unsqueeze(0), mat.unsqueeze(0)), 0)
            else:
                mat = self.prob_matrix.T @ attr_mat[-1]
                attr_mat = torch.cat((attr_mat, mat.unsqueeze(0)), 0)

        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_mat.T)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = torch.sigmoid(self.fcs[-1](self.dropout(layer_inner))) # It must be sigmoid.
        return res

    def loss(self, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss


class DiffusionPropagate(nn.Module):
    def __init__(self, prob_matrix, niter):
        super(DiffusionPropagate, self).__init__()

        self.niter = niter

        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()

        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))

    def forward(self, preds, seed_idx):
        # import ipdb; ipdb.set_trace()
        device = preds.device
        # prop_preds = torch.ones((preds.shape[0], preds.shape[1])).to(device)


        for i in range(preds.shape[0]):
            prop_pred = preds[i]
            for j in range(self.niter):
                P2 = self.prob_matrix.T * prop_pred.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                prop_pred = torch.ones((self.prob_matrix.shape[0],)).to(device) - torch.prod(P3, dim=1)
                prop_pred[seed_idx[seed_idx[:,0] == i][:, 1]] = 1 # commented in SLVAE
                prop_pred = prop_pred.unsqueeze(0)
            if i == 0:
                prop_preds = prop_pred
            else:
                prop_preds = torch.cat((prop_preds, prop_pred), 0)

        return prop_preds


class InverseModel(nn.Module):
    def __init__(self, vae_model: nn.Module, gnn_model: nn.Module, propagate: nn.Module):
        super(InverseModel, self).__init__()

        self.vae_model = vae_model
        self.gnn_model = gnn_model
        self.propagate = propagate

        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))

    def forward(self, seed_vec, static_features, seed_features):
        device = next(self.gnn_model.parameters()).device
        seed_idx = torch.LongTensor(np.argwhere(seed_vec.cpu().detach().numpy() == 1)).to(device)

        seed_hat, mean, log_var = self.vae_model(seed_vec, static_features, seed_features)
        predictions = self.gnn_model(seed_hat[0].squeeze())
        predictions = self.propagate(predictions, seed_idx)

        return seed_hat, mean, log_var, predictions

    def loss(self, x, x_hat, mean, log_var, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
        #reproduction_loss = F.mse_loss(x_hat, x)
        KLD = -0.5*torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        total_loss = forward_loss+reproduction_loss+ 1e-3 * KLD
        return KLD, reproduction_loss, forward_loss, total_loss


class ForwardModel(nn.Module):
    def __init__(self, gnn_model: nn.Module, propagate: nn.Module):
        super(ForwardModel, self).__init__()
        self.gnn_model = gnn_model
        self.propagate = propagate
        self.relu = nn.ReLU(inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))

    def forward(self, seed_vec):
        device = next(self.gnn_model.parameters()).device

        # seed_idx = torch.LongTensor(np.argwhere(seed_vec.cpu().detach().numpy() == 1)).to(device)
        seed_idx = torch.LongTensor(np.argwhere(seed_vec.cpu().detach().numpy() >args.seed_threshold)).to(device)
        # seed_idx = (seed_vec == 1).nonzero(as_tuple=False) # Used in SLVAE

        predictions = self.gnn_model(seed_vec) # last layer is sigmoid
        # predictions = self.gnn_model(seed_vec).squeeze()  # last layer is sigmoid
        predictions = self.propagate(predictions, seed_idx)

        predictions = (predictions + seed_vec)/2 # commented in SLVAE

        # predictions = self.relu(predictions) # Used in SLVAE No impact (relu or selu)
        # predictions = torch.sigmoid(predictions) # damaging

        return predictions

    def loss(self, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss

### Inference part
def x_hat_initialization(model_proj, model_rec,
                         inf_proj_idx, nodes_name_G_recived, seed_name_received,
                         device, lossFn,
                         x_hat, x, y, x_rec, y_rec,
                         f_z_bar, test_id, input_optimizer, epochs=100,
                         threshold=0.55):

    initial_x, initial_x_performance = [], []

    for epoch in range(epochs):
        input_optimizer.zero_grad()
        y_hat = model_proj(x_hat.squeeze())
        seed_vector_rec = torch.zeros(y_rec.shape)
        for p_i in range(y_hat.shape[0]):  # batch size
            infect_proj = y_hat[p_i, :].cpu().clone().detach()
            seed_value = infect_proj[inf_proj_idx]
            seed_vector_rec[p_i, :][
                np.where(
                    nodes_name_G_recived.reshape(nodes_name_G_recived.size, 1) == seed_name_received
                )[0]] = seed_value[np.where(
                nodes_name_G_recived.reshape(nodes_name_G_recived.size, 1) == seed_name_received)[1]]
        x_hat_rec = seed_vector_rec.to(device)
        y_hat_rec = model_rec(x_hat_rec)

        loss, pdf_loss = lossFn(y, y_hat, y_rec, y_hat_rec, x_hat, f_z_bar)

        x_pred = x_hat.clone().cpu().detach()
        # x = x_true.cpu().detach().numpy()

        x_pred[x_pred > threshold] = 1
        x_pred[x_pred != 1] = 0

        x_true = x.cpu().detach()
        seed_original = x_true.squeeze().cpu().detach().numpy().flatten()
        seed_predicted = x_pred.squeeze().cpu().detach().numpy().flatten()

        accuracy = accuracy_score(seed_original, seed_predicted)
        precision = precision_score(seed_original, seed_predicted, zero_division=0)
        recall = recall_score(seed_original, seed_predicted, zero_division=0)
        f1 = f1_score(seed_original, seed_predicted, zero_division=0)

        #         precision = precision_score(x[0], x_pred[0])
        #         recall = recall_score(x[0], x_pred[0])
        #         f1 = f1_score(x[0], x_pred[0])

        loss.backward()
        input_optimizer.step()

        with torch.no_grad():
            x_hat.clamp_(0, 1)

        initial_x.append(x_hat)
        # initial_x_performance.append([accuracy, precision, recall, f1])
        initial_x_performance.append(f1)

    return initial_x, initial_x_performance