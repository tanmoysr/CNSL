#%% Workspace Setup
import torch
from torch.optim import Adam, SGD
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn

# Customized Libraries
import loss_functions
import models
import gat
import utilities as utils
import scipy.sparse as sp
from configuration import args

# device = 'cpu'
print(torch.version.cuda)
print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1
print(args)

#%% File Names
dataset_file = '{}{}{}2{}_{}_{}.SG'.format(args.data_loc, args.dataset, args.diffusion_model_proj,
                                    args.diffusion_model_rec, str(10*args.seed_rate), args.sample)
epoch_log_file = args.model_loc + 'inference_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
# VAE
feature_representer_chk_file = args.model_loc + 'feature_representer_train_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.ckpt'
vae_chk_file = args.model_loc + 'VAE_train_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.ckpt'
epoch_log_file_vae = args.model_loc + 'VAE_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'

# Diffusion Projection
frwrd_proj_file = args.model_loc + 'forward_model_proj_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.ckpt'
epoch_log_file_diffProj = args.model_loc + 'diffProj_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'

# Diffusion Recipient
frwrd_rec_file = args.model_loc + 'forward_model_rec_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.ckpt'
epoch_log_file_diffRec = args.model_loc + 'diffRec_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
#%% Loding Dataset
with open(dataset_file, 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs, prob_matrix, static_features_pre = graph['adj_proj'], graph['inverse_pairs_proj'], graph['prob_proj'], graph['static_features_proj']
adj_rec, inverse_pairs_rec, prob_matrix_rec = graph['adj_received'], graph['inverse_pairs_received'], graph['prob_received']
G_proj, proj_nodes, G_received, receipient_nodes = graph['original_graph_proj'], graph['proj_nodes'], graph['original_graph_received'], graph['receipient_nodes']

nodes_name_G_proj = np.array(list(G_proj.nodes()))
nodes_name_G_recived = np.array(list(G_received.nodes()))
# nodes_name_G_recived = nodes_name_G_recived.astype(str)

inf_proj_idx = []
for i in nodes_name_G_proj:
    inf_proj_idx.extend(np.where(proj_nodes == i)[0].tolist())
seed_name_received = receipient_nodes[inf_proj_idx]

seed_rec_idx = []
for i in nodes_name_G_recived:
    seed_rec_idx.extend(np.where(seed_name_received == i)[0].tolist())
seed_name_received = receipient_nodes[inf_proj_idx]

## Standardization
# std_scalar = StandardScaler()
# static_features_std = torch.from_numpy(std_scalar.fit_transform(static_features_pre))
static_features_std = torch.from_numpy(preprocessing.normalize(static_features_pre))

#%% Splitting data
batch_size = args.batchSizeInfer
random_seed = 42
if inverse_pairs.shape[0] == inverse_pairs_rec.shape[0]:
    indices = list(range(inverse_pairs.shape[0]))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    # train_indices, test_indices = indices[:len(inverse_pairs)-batch_size], indices[batch_size:]
    train_indices, test_indices = indices[:int(.8*len(indices))], indices[int(.8*len(indices)):]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_set = torch.utils.data.DataLoader(inverse_pairs, batch_size=batch_size,
                                           sampler=train_sampler,drop_last=True)
    test_set = torch.utils.data.DataLoader(inverse_pairs, batch_size=batch_size,
                                                sampler=test_sampler,drop_last=True)
    train_set_rec = torch.utils.data.DataLoader(inverse_pairs_rec, batch_size=batch_size,
                                           sampler=train_sampler,drop_last=True)
    test_set_rec = torch.utils.data.DataLoader(inverse_pairs_rec, batch_size=batch_size,
                                                sampler=test_sampler,drop_last=True)
else:
    print("Data mismatched")
static_features = static_features_std.to(device)

#%% Model Setup
torch.manual_seed(random_seed)
feature_representer_layer = 2
feature_representer_hidden = int(32*1)
encoder_seed_hidden = int(32*1)
encoder_seed_latent = int(16*1)
encoder_features_hidden = int(32*1)
encoder_features_latent = int(16*1)
encoder_seed_features_hidden = int(32*1)
encoder_seed_features_latent = int(16*1)

## VAE
feature_representer = models.MLP(num_layers=feature_representer_layer, input_dim=static_features.shape[1],
                          hidden_dim=feature_representer_hidden, output_dim=static_features.shape[1]).to(device)
encoder_seed = models.Encoder(input_dim= 1, hidden_dim=encoder_seed_hidden, latent_dim=encoder_seed_latent)
encoder_features = models.Encoder(input_dim= static_features.shape[1], hidden_dim=encoder_features_hidden, latent_dim=encoder_features_latent)
encoder_seed_features = models.Encoder(input_dim= args.input_feature_dim+1, hidden_dim=encoder_seed_features_hidden, latent_dim=encoder_seed_features_latent)  # 16 for social2colocation, 5 for git2stack

decoder_seed = models.Decoder(input_dim = encoder_seed_latent+encoder_seed_features_latent, latent_dim=32, hidden_dim =16, output_dim = 1, seed_pred = True)
decoder_features = models.Decoder(input_dim = encoder_features_latent+encoder_seed_features_latent, latent_dim=32, hidden_dim =16,
                           output_dim = static_features.shape[1], seed_pred = False)
decoder_seed_features = models.Decoder(input_dim = encoder_seed_features_latent, latent_dim=32, hidden_dim =16, output_dim = 16, seed_pred = True, seed_plus = True) # 16 for social2colocation, 5 for git2stack

vae_model = models.VAEModelComb(encoder_seed, encoder_features, encoder_seed_features,
                     decoder_seed, decoder_features, decoder_seed_features).to(device)
feature_representer = feature_representer.to(device)
vae_model = vae_model.to(device)

## Diffusion Projection
gnn_model_proj = models.GNNModel(input_dim=5, # hop
                     hiddenunits=[64, 64], # 64,64
                     num_classes=1,
                     prob_matrix=prob_matrix)
propagate_proj = models.DiffusionPropagate(prob_matrix, niter=2)
forward_model_proj = models.ForwardModel(gnn_model_proj, propagate_proj).to(device)
forward_model_proj = forward_model_proj.to(device)
forward_model_proj = forward_model_proj.to(device)

## Diffusion Recipient
# gnn_model_rec = models.GNNModel(input_dim=5, # hop
#                      hiddenunits=[64, 64],
#                      num_classes=1,
#                      prob_matrix=prob_matrix_rec)
gnn_model_rec = models.GNNModel(input_dim=3, # hop
                     hiddenunits=[64, 64],
                     num_classes=1,
                     prob_matrix=prob_matrix_rec,
                    bias=False, drop_prob=0.8)
propagate_rec = models.DiffusionPropagate(prob_matrix_rec, niter=2)
forward_model_rec = models.ForwardModel(gnn_model_rec, propagate_rec).to(device)
forward_model_rec = forward_model_rec.to(device)

#%% loading VAE models
checkpoint_feature_representer = torch.load(feature_representer_chk_file)
feature_representer.load_state_dict(checkpoint_feature_representer)
feature_representer.to(device)
feature_representer.eval()
checkpoint_vae = torch.load(vae_chk_file)
vae_model.load_state_dict(checkpoint_vae)
vae_model.to(device)
vae_model.eval()

#%% loading diffusion projection model
checkpoint_forward_model_proj = torch.load(frwrd_proj_file)
forward_model_proj.load_state_dict(checkpoint_forward_model_proj)
forward_model_proj.to(device)
forward_model_proj.eval()

#%% loading diffusion recipient model
checkpoint_forward_model_rec = torch.load(frwrd_rec_file)
forward_model_rec.load_state_dict(checkpoint_forward_model_rec)
forward_model_rec.to(device)
forward_model_rec.eval()

#%% Inference Initialization
print("Inference Initialization Started ...")
train_merge = torch.tensor(np.vstack(np.array([t.numpy() for t in list(np.array(list(enumerate(train_set)))[:,1])])))
train_x = train_merge[:, :, 0].float().to(device)
train_y = train_merge[:, :, 1].float().to(device)

with torch.no_grad():
    x_features = torch.cat((train_x.unsqueeze(-1), feature_representer(static_features).repeat(train_x.shape[0], 1, 1)), -1)
    # x_features.shape
    sf2 = [train_x.unsqueeze(-1), static_features.repeat(train_x.shape[0], 1, 1), x_features]
    sf2_hat, mean, log_var = vae_model(train_x.unsqueeze(-1), static_features.repeat(train_x.shape[0], 1, 1), x_features)

    ## Getting latent distribution
    mean_sd, log_var_sd = vae_model.Encoder_sd(train_x.unsqueeze(-1))
    mean_ft, log_var_ft = vae_model.Encoder_ft(static_features.repeat(train_x.shape[0], 1, 1))
    mean_sd_ft, log_var_sd_ft = vae_model.Encoder_sd_ft(x_features)

    z_sd = vae_model.reparameterization(mean_sd, log_var_sd)  # takes exponential function (log var -> var)
    z_ft = vae_model.reparameterization(mean_ft, log_var_ft)
    z_sd_ft = vae_model.reparameterization(mean_sd_ft, log_var_sd_ft)

    z_sd_plus = torch.cat((z_sd, z_sd_ft), -1)
    z_ft_plus = torch.cat((z_ft, z_sd_ft), -1)

    z_sd_plus_bar = torch.mean(z_sd_plus, dim=0)

    f_z_all = vae_model.Decoder_sd(z_sd_plus)
    f_z_bar = vae_model.Decoder_sd(z_sd_plus_bar)

    x_hat = torch.sigmoid(torch.randn(f_z_all[:batch_size].shape)).to(device)
    x_hat = f_z_bar.repeat(batch_size, 1, 1).to(device)

x_hat.requires_grad = True

#%% Final Inference
print("Final Inference Started ...")
optimizer_input = Adam([x_hat], lr=args.lr_Infer, weight_decay=args.wd_Infer)

sample_number = len(test_indices)

epoch_dict = {'loss': [], 'forward_loss': [],
                  'seed accuracy': [], 'seed precision': [], 'seed precision k': [], 'seed recall': [],
                'seed_f1': [], 'seed_auc': [], 'auc_proj': [], 'auc_rec': [], 'time': []}
for epoch in range(args.numEpochInfer):

    epoch_tic = time.perf_counter()
    loss_overall = 0
    forward_overall = 0
    seed_accuracy_all = 0
    seed_precision_all = 0
    seed_precision_k_all = 0
    seed_recall_all = 0
    seed_f1_all = 0
    seed_auc_all = 0
    auc_all_proj = 0
    auc_all_rec = 0

    optimizer_input.zero_grad()

    dataloader_iterator = iter(test_set_rec)
    for batch_idx, data_pair in enumerate(test_set):
        try:
            data_pair_rec = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(test_set_rec)
            data_pair_rec = next(dataloader_iterator)

        x = data_pair[:, :, 0].unsqueeze(2).float().to(device)
        x_true = x.cpu().detach()
        y = data_pair[:, :, 1].to(device)
        x_rec = data_pair_rec[:, :, 0].unsqueeze(2).float().to(device)
        x_true_rec = x_rec.cpu().detach()
        y_rec = data_pair_rec[:, :, 1].to(device)

        y_hat = forward_model_proj(x_hat.squeeze())
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
        y_hat_rec = forward_model_rec(x_hat_rec)
        loss, forward_loss = loss_functions.loss_inverse(y_rec, y_hat_rec, x_hat)
        loss_overall += loss.item() * x_hat.size(0)
        forward_overall += forward_loss.item() * x_hat.size(0)

        loss.backward()
        optimizer_input.step()

        ## Performance
        x_pred = x_hat.cpu().detach()
        x_pred[x_pred > args.seed_threshold] = 1
        x_pred[x_pred != 1] = 0

        seed_original = x_true.squeeze().cpu().detach().numpy().flatten()
        seed_hat = x_hat.squeeze().cpu().detach().numpy().flatten()
        seed_predicted = x_pred.squeeze().cpu().detach().numpy().flatten()

        seed_accuracy_all += accuracy_score(seed_original.astype(int), seed_predicted.astype(int))
        seed_precision_all += precision_score(seed_original.astype(int), seed_predicted.astype(int), zero_division=0)
        seed_precision_k_all +=utils.precision_at_k_score(seed_original.astype(int), seed_hat)
        seed_recall_all += recall_score(seed_original.astype(int), seed_predicted.astype(int), zero_division=0)
        seed_f1_all += f1_score(seed_original.astype(int), seed_predicted.astype(int))
        seed_auc_all += roc_auc_score(seed_original, seed_hat)

        infected_original_proj = y.squeeze().cpu().detach().numpy().flatten()
        infected_predicted_proj = y_hat.squeeze().cpu().detach().numpy().flatten()
        auc_all_proj += roc_auc_score(infected_original_proj, infected_predicted_proj)

        infected_original_rec = y_rec.squeeze().cpu().detach().numpy().flatten()
        infected_predicted_rec = y_hat_rec.squeeze().cpu().detach().numpy().flatten()
        auc_all_rec += roc_auc_score(infected_original_rec, infected_predicted_rec)

    args.seed_threshold = utils.find_bestThreshold(seed_original, seed_hat)
    epoch_toc = time.perf_counter()
    epoch_time = epoch_toc - epoch_tic
    epoch_dict['loss'].append(loss_overall / (batch_idx + 1))
    epoch_dict['forward_loss'].append(forward_overall / (batch_idx + 1))
    epoch_dict['seed accuracy'].append(seed_accuracy_all / (batch_idx + 1))
    epoch_dict['seed precision'].append(seed_precision_all / (batch_idx + 1))
    epoch_dict['seed precision k'].append(seed_precision_k_all / (batch_idx + 1))
    epoch_dict['seed recall'].append(seed_recall_all / (batch_idx + 1))
    epoch_dict['seed_f1'].append(seed_f1_all / (batch_idx + 1))
    epoch_dict['seed_auc'].append(seed_auc_all / (batch_idx + 1))
    epoch_dict['auc_proj'].append(auc_all_proj / (batch_idx + 1))
    epoch_dict['auc_rec'].append(auc_all_rec / (batch_idx + 1))
    epoch_dict['time'].append(epoch_time)
    print("Epoch: {}".format(epoch + 1),
          "\tLoss: {:.4f}".format(loss_overall / (batch_idx + 1)),
          "\tForward Loss: {:.4f}".format(forward_overall / (batch_idx + 1)),
          # "\tSeed Accuracy: {:.4f}".format(seed_accuracy_all / (batch_idx + 1)),
          "\tSeed Precision: {:.4f}".format(seed_precision_all / (batch_idx + 1)),
          "\tSeed Precision K : {:.4f}".format(seed_precision_k_all / (batch_idx + 1)),
          "\tSeed Recall: {:.4f}".format(seed_recall_all / (batch_idx + 1)),
          "\tSeed F1: {:.4f}".format(seed_f1_all / (batch_idx + 1)),
          "\tSeed AUC: {:.4f}".format(seed_auc_all / (batch_idx + 1)),
          "\tAUC Proj: {:.4f}".format(auc_all_proj / (batch_idx + 1)),
          "\tAUC Rec: {:.4f}".format(auc_all_rec / (batch_idx + 1)),
          "\tTime Taken: {:.6f}".format(epoch_time)
          )
epoch_df = pd.DataFrame.from_dict(epoch_dict)
epoch_df.index = epoch_df.index.rename('epochs')
epoch_df.to_csv(epoch_log_file)
