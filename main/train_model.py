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

#%% Train setup beyond configuration files
vae_train_need = True
diff1_train_need = True
diff2_train_need = True
all_train_need = True
#%% File Names
dataset_file = '{}{}{}2{}_{}_{}.SG'.format(args.data_loc, args.dataset, args.diffusion_model_proj,
                                    args.diffusion_model_rec, str(10*args.seed_rate), args.sample)

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
# train together
epoch_log_file = args.model_loc + 'train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'

#%% Loding Dataset
with open(dataset_file, 'rb') as f:
    graph = pickle.load(f)

adj, inverse_pairs, prob_matrix, static_features_pre = graph['adj_proj'], graph['inverse_pairs_proj'], graph['prob_proj'], graph['static_features_proj']
adj_rec, inverse_pairs_rec, prob_matrix_rec = graph['adj_received'], graph['inverse_pairs_received'], graph['prob_received']
G_proj, proj_nodes, G_received, receipient_nodes = graph['original_graph_proj'], graph['proj_nodes'], graph['original_graph_received'], graph['receipient_nodes']

nodes_name_G_proj = np.array(list(G_proj.nodes()))
nodes_name_G_recived = np.array(list(G_received.nodes()))

# String conversion
receipient_nodes = receipient_nodes.astype(str)
nodes_name_G_recived = nodes_name_G_recived.astype(str)

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
batch_size = args.batchSize
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

#%% Train VAE
optimizer_vae = Adam([{'params': feature_representer.parameters()}, {'params': vae_model.parameters()}],
                     lr=args.lr_VAE, weight_decay=args.wd_VAE)

if vae_train_need:
    feature_representer.train()
    vae_model.train()
    sample_number = len(train_indices)
    epoch_dict_vae = {'reconstruction': [], 'kld': [], 'total_vae': [],
                      'accuracy': [], 'precision': [], 'recall': [], 'time': []}

    ## Training
    for epoch in range(args.numEpochVAE):

        epoch_tic = time.perf_counter()
        re_overall = 0
        kld_overall = 0
        total_loss_vae = 0

        seed_accuracy_all = 0
        seed_precision_all = 0
        seed_recall_all = 0

        dataloader_iterator = iter(train_set_rec)
        for batch_idx, data_pair in enumerate(train_set):
            try:
                data_pair_rec = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_set_rec)
                data_pair_rec = next(dataloader_iterator)

            optimizer_vae.zero_grad()

            x = data_pair[:, :, 0].unsqueeze(2).float().to(device)
            # x_features = torch.cat((x, static_features), 1)
            x_features = torch.cat((x, feature_representer(static_features).repeat(batch_size, 1, 1)), -1)
            x_true = x.cpu().detach()
            sf2 = [x, static_features.repeat(batch_size, 1, 1), x_features]

            sf2_hat, mean, log_var = vae_model(x, static_features.repeat(batch_size, 1, 1), x_features)
            x_hat, static_features_hat, x_features_hat = sf2_hat

            re_loss, kld, loss_vae = loss_functions.loss_vae(sf2, sf2_hat, mean, log_var)
            re_overall += re_loss.item() * x_hat.size(0)
            kld_overall += kld.item() * x_hat.size(0)
            # for k in kld:
            #     kld_overall += k.item() * x_hat.size(0)
            total_loss_vae += loss_vae.item() * x_hat.size(0)

            loss_vae.backward()
            optimizer_vae.step()

            ## Performance
            x_pred = x_hat.cpu().detach()
            x_pred[x_pred > args.seed_threshold] = 1  # 0.01
            x_pred[x_pred != 1] = 0

            seed_original = x_true.squeeze().cpu().detach().numpy().flatten()
            seed_predicted = x_pred.squeeze().cpu().detach().numpy().flatten()

            seed_accuracy_all += accuracy_score(seed_original, seed_predicted)
            seed_precision_all += precision_score(seed_original, seed_predicted, zero_division=0)
            seed_recall_all += recall_score(seed_original, seed_predicted, zero_division=0)

        epoch_toc = time.perf_counter()
        epoch_time = epoch_toc - epoch_tic
        epoch_dict_vae['reconstruction'].append(re_overall / (batch_idx + 1))
        epoch_dict_vae['kld'].append(kld_overall / (batch_idx + 1))
        epoch_dict_vae['total_vae'].append(total_loss_vae / (batch_idx + 1))
        epoch_dict_vae['accuracy'].append(seed_accuracy_all / (batch_idx + 1))
        epoch_dict_vae['precision'].append(seed_precision_all / (batch_idx + 1))
        epoch_dict_vae['recall'].append(seed_recall_all / (batch_idx + 1))
        epoch_dict_vae['time'].append(epoch_time)
        print("VAE Epoch {}".format(epoch + 1),
              "\tReconstruction: {:.4f}".format(re_overall / (batch_idx + 1)),
              "\tKLD: {:.4f}".format(kld_overall / (batch_idx + 1)),
              "\tTotal: {:.4f}".format(total_loss_vae / (batch_idx + 1)),
              "\tAccuracy: {:.4f}".format(seed_accuracy_all / (batch_idx + 1)),
              "\tPrecision: {:.4f}".format(seed_precision_all / (batch_idx + 1)),
              "\tRecall: {:.4f}".format(seed_recall_all / (batch_idx + 1)),
              "\tTime Taken: {:.6f}".format(epoch_time)
              )
    #%% saving  VAE models
    torch.save(feature_representer.state_dict(), feature_representer_chk_file)
    print(' Feature Representaer Model saved')

    torch.save(vae_model.state_dict(), vae_chk_file)
    print(' VAE Model saved')

    epoch_df_vae = pd.DataFrame.from_dict(epoch_dict_vae)
    epoch_df_vae.index = epoch_df_vae.index.rename('epochs')
    epoch_df_vae.to_csv(epoch_log_file_vae)

#%% loading VAE models
checkpoint_feature_representer = torch.load(feature_representer_chk_file)
feature_representer.load_state_dict(checkpoint_feature_representer)
feature_representer.to(device)
feature_representer.eval()
checkpoint_vae = torch.load(vae_chk_file)
vae_model.load_state_dict(checkpoint_vae)
vae_model.to(device)
vae_model.eval()

#%% Train Diffusion Projection
for param in vae_model.parameters():
    param.requires_grad = False
for param in feature_representer.parameters():
    param.requires_grad = False

optimizer_diffProj = Adam([{'params': forward_model_proj.parameters()}], lr=args.lr_DiffProj,
                          weight_decay=args.wd_DiffProj)
## Training
if diff1_train_need:
    forward_model_proj.train()
    sample_number = len(train_indices)
    epoch_dict_diffProj = {'total_diffProj': [], 'infect_precision': [], 'infect_recall': [], 'infect_accuracy': [],
                           'time': []}
    for epoch in range(args.numEpochDiffProj):

        epoch_tic = time.perf_counter()

        total_loss_diffProj = 0

        infect_accuracy_all = 0
        infect_precision_all = 0
        infect_recall_all = 0

        dataloader_iterator = iter(train_set_rec)
        for batch_idx, data_pair in enumerate(train_set):
            try:
                data_pair_rec = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_set_rec)
                data_pair_rec = next(dataloader_iterator)

            optimizer_diffProj.zero_grad()

            x = data_pair[:, :, 0].unsqueeze(2).float().to(device)
            y = data_pair[:, :, 1].to(device)
            # x_features = torch.cat((x, static_features), 1)
            x_features = torch.cat((x, feature_representer(static_features).repeat(batch_size, 1, 1)), -1)
            x_true = x.cpu().detach()
            sf2 = [x, static_features.repeat(batch_size, 1, 1), x_features]

            # x_rec = data_pair_rec[:, :, 0].unsqueeze(2).float().to(device)
            # x_true_rec = x_rec.cpu().detach()
            # y_rec = data_pair_rec[:, :, 1].to(device)

            sf2_hat, mean, log_var = vae_model(x, static_features.repeat(batch_size, 1, 1), x_features)
            x_hat, static_features_hat, x_features_hat = sf2_hat
            y_hat = forward_model_proj(x_hat.squeeze())

            loss_proj = loss_functions.loss_proj(y_hat, y)
            total_loss_diffProj += loss_proj.item() * x_hat.size(0)
            loss_proj.backward()
            optimizer_diffProj.step()

            y_pred = y_hat.cpu().detach()
            y_pred[y_pred > args.seed_threshold] = 1  # 0.01
            y_pred[y_pred != 1] = 0

            infected_original_proj = y.squeeze().cpu().detach().numpy().flatten()
            infected_predicted_proj = y_pred.squeeze().cpu().detach().numpy().flatten()

            infect_accuracy_all += accuracy_score(infected_original_proj, infected_predicted_proj)
            infect_precision_all += precision_score(infected_original_proj, infected_predicted_proj, zero_division=0)
            infect_recall_all += recall_score(infected_original_proj, infected_predicted_proj, zero_division=0)

        epoch_toc = time.perf_counter()
        epoch_time = epoch_toc - epoch_tic
        epoch_dict_diffProj['total_diffProj'].append(total_loss_diffProj / (batch_idx + 1))
        epoch_dict_diffProj['infect_accuracy'].append(infect_accuracy_all / (batch_idx + 1))
        epoch_dict_diffProj['infect_precision'].append(infect_precision_all / (batch_idx + 1))
        epoch_dict_diffProj['infect_recall'].append(infect_recall_all / (batch_idx + 1))
        epoch_dict_diffProj['time'].append(epoch_time)
        print("Diffusion Projection Epoch {}".format(epoch + 1),
              "\tTotal: {:.4f}".format(total_loss_diffProj / (batch_idx + 1)),
              "\tAccuracy: {:.4f}".format(infect_accuracy_all / (batch_idx + 1)),
              "\tPrecision: {:.4f}".format(infect_precision_all / (batch_idx + 1)),
              "\tRecall: {:.4f}".format(infect_recall_all / (batch_idx + 1)),
              "\tTime Taken: {:.6f}".format(epoch_time)
              )

    #%% saving diffusion projection models
    torch.save(forward_model_proj.state_dict(), frwrd_proj_file)
    print('Diffusion model for projection graph saved')

    epoch_df_diffProj = pd.DataFrame.from_dict(epoch_dict_diffProj)
    epoch_df_diffProj.index = epoch_df_diffProj.index.rename('epochs')
    epoch_df_diffProj.to_csv(epoch_log_file_diffProj)

#%% loading diffusion projection model
checkpoint_forward_model_proj = torch.load(frwrd_proj_file)
forward_model_proj.load_state_dict(checkpoint_forward_model_proj)
forward_model_proj.to(device)
forward_model_proj.eval()
#%% Train Diffusion Recipient
for param in vae_model.parameters():
    param.requires_grad = False
for param in feature_representer.parameters():
    param.requires_grad = False
for param in forward_model_proj.parameters():
    param.requires_grad = False

optimizer_diffRec = Adam([{'params': forward_model_rec.parameters()}],lr=args.lr_DiffRec, weight_decay = args.wd_DiffRec)
## Training
if diff2_train_need:
    forward_model_rec.train()
    sample_number = len(train_indices)
    epoch_dict_diffRec = {'total_diffRec': [], 'infect_precision': [], 'infect_recall': [], 'infect_accuracy': [],
                          'time': []}
    for epoch in range(args.numEpochDiffRec):

        epoch_tic = time.perf_counter()

        total_loss_diffRec = 0

        infect_precision_all_rec = 0
        infect_recall_all_rec = 0
        infect_accuracy_all_rec = 0

        dataloader_iterator = iter(train_set_rec)
        for batch_idx, data_pair in enumerate(train_set):
            try:
                data_pair_rec = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_set_rec)
                data_pair_rec = next(dataloader_iterator)

            optimizer_diffRec.zero_grad()

            if args.select_proj2recMap:
                x = data_pair[:, :, 0].unsqueeze(2).float().to(device)
                y = data_pair[:, :, 1].to(device)
                # x_features = torch.cat((x, static_features), 1)
                x_features = torch.cat((x, feature_representer(static_features).repeat(batch_size, 1, 1)), -1)
                x_true = x.cpu().detach()
                sf2 = [x, static_features.repeat(batch_size, 1, 1), x_features]

                x_rec = data_pair_rec[:, :, 0].unsqueeze(2).float().to(device)
                x_true_rec = x_rec.cpu().detach()
                y_rec = data_pair_rec[:, :, 1].to(device)

                sf2_hat, mean, log_var = vae_model(x, static_features.repeat(batch_size, 1, 1), x_features)
                x_hat, static_features_hat, x_features_hat = sf2_hat
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
            else:
                x = data_pair[:, :, 0].unsqueeze(2).float().to(device)
                y = data_pair[:, :, 1].to(device)
                # x_features = torch.cat((x, static_features), 1)
                x_features = torch.cat((x, feature_representer(static_features).repeat(batch_size, 1, 1)), -1)
                x_true = x.cpu().detach()
                sf2 = [x, static_features.repeat(batch_size, 1, 1), x_features]

                x_rec = data_pair_rec[:, :, 0].unsqueeze(2).float().to(device)
                x_true_rec = x_rec.cpu().detach()
                y_rec = data_pair_rec[:, :, 1].to(device)

                sf2_hat, mean, log_var = vae_model(x, static_features.repeat(batch_size, 1, 1), x_features)
                x_hat, static_features_hat, x_features_hat = sf2_hat
                y_hat = forward_model_proj(x_hat.squeeze())
                seed_vector_rec = x_rec.squeeze(2)

            x_hat_rec = seed_vector_rec.to(device)
            y_hat_rec = forward_model_rec(x_hat_rec)

            loss_rec = loss_functions.loss_rec(y_hat_rec, y_rec)
            total_loss_diffRec += loss_rec.item() * x_hat.size(0)
            loss_rec.backward()
            optimizer_diffRec.step()

            y_pred_rec = y_hat_rec.cpu().detach()
            y_pred_rec[y_pred_rec > args.seed_threshold] = 1  # 0.01
            y_pred_rec[y_pred_rec != 1] = 0

            infected_original_rec = y_rec.squeeze().cpu().detach().numpy().flatten()
            infected_predicted_rec = y_pred_rec.squeeze().cpu().detach().numpy().flatten()

            infect_accuracy_all_rec += accuracy_score(infected_original_rec.astype(int), infected_predicted_rec.astype(int))
            infect_precision_all_rec += precision_score(infected_original_rec.astype(int), infected_predicted_rec.astype(int), zero_division=0)
            infect_recall_all_rec += recall_score(infected_original_rec.astype(int), infected_predicted_rec.astype(int), zero_division=0)

        epoch_toc = time.perf_counter()
        epoch_time = epoch_toc - epoch_tic
        epoch_dict_diffRec['total_diffRec'].append(total_loss_diffRec / (batch_idx + 1))
        epoch_dict_diffRec['infect_accuracy'].append(infect_accuracy_all_rec / (batch_idx + 1))
        epoch_dict_diffRec['infect_precision'].append(infect_precision_all_rec / (batch_idx + 1))
        epoch_dict_diffRec['infect_recall'].append(infect_recall_all_rec / (batch_idx + 1))
        epoch_dict_diffRec['time'].append(epoch_time)
        print("Diffusion Receipient Epoch {}".format(epoch + 1),
              "\tTotal: {:.4f}".format(total_loss_diffRec / (batch_idx + 1)),
              "\tAccuracy: {:.4f}".format(infect_accuracy_all_rec / (batch_idx + 1)),
              "\tPrecision: {:.4f}".format(infect_precision_all_rec / (batch_idx + 1)),
              "\tRecall: {:.4f}".format(infect_recall_all_rec / (batch_idx + 1)),
              "\tTime Taken: {:.6f}".format(epoch_time)
              )
    #%% saving diffusion recipient models
    torch.save(forward_model_rec.state_dict(), frwrd_rec_file)
    print('Diffusion model for receipient graph saved')

    epoch_df_diffRec = pd.DataFrame.from_dict(epoch_dict_diffRec)
    epoch_df_diffRec.index = epoch_df_diffRec.index.rename('epochs')
    epoch_df_diffRec.to_csv(epoch_log_file_diffRec)

#%% loading diffusion recipient model
checkpoint_forward_model_rec = torch.load(frwrd_rec_file)
forward_model_rec.load_state_dict(checkpoint_forward_model_rec)
forward_model_rec.to(device)
forward_model_rec.eval()

#%% train together
if all_train_need:
    sample_number = len(train_indices)
    epoch_dict = {'reconstruction': [], 'kld': [], 'total': [],
                  'seed accuracy': [], 'seed precision': [], 'seed recall': [],
                  'accuracy': [], 'precision': [], 'recall': [],
                  'accuracy_rec': [], 'precision_rec': [], 'recall_rec': [],
                  'time': []}
    for param in vae_model.parameters():
        param.requires_grad = True
    for param in feature_representer.parameters():
        param.requires_grad = True
    for param in forward_model_proj.parameters():
        param.requires_grad = True
    for param in forward_model_rec.parameters():
        param.requires_grad = True
    feature_representer.train()
    vae_model.train()
    forward_model_proj.train()
    forward_model_rec.train()
    optimizer = utils.MultipleOptimizer(optimizer_vae, optimizer_diffProj, optimizer_diffRec)
    for epoch in range(args.numEpoch):

        epoch_tic = time.perf_counter()
        re_overall = 0
        kld_overall = 0
        total_overall = 0

        seed_accuracy_all = 0
        seed_precision_all = 0
        seed_recall_all = 0

        precision_all = 0
        recall_all = 0
        accuracy_all = 0
        precision_all_rec = 0
        recall_all_rec = 0
        accuracy_all_rec = 0

        dataloader_iterator = iter(train_set_rec)
        for batch_idx, data_pair in enumerate(train_set):
            try:
                data_pair_rec = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_set_rec)
                data_pair_rec = next(dataloader_iterator)


            optimizer.zero_grad()

            x = data_pair[:, :, 0].unsqueeze(2).float().to(device)
            y = data_pair[:, :, 1].to(device)
            # x_features = torch.cat((x, static_features), 1)
            x_features = torch.cat((x, feature_representer(static_features).repeat(batch_size, 1, 1)), 2)
            x_true = x.cpu().detach()
            sf2 = [x, static_features.repeat(batch_size, 1, 1), x_features]

            x_rec = data_pair_rec[:, :, 0].unsqueeze(2).float().to(device)
            x_true_rec = x_rec.cpu().detach()
            y_rec = data_pair_rec[:, :, 1].to(device)

            sf2_hat, mean, log_var = vae_model(x, static_features.repeat(batch_size, 1, 1), x_features)
            x_hat, static_features_hat, x_features_hat = sf2_hat
            y_hat = forward_model_proj(x_hat.squeeze())
            # x_hat_det = x_hat.detach()
            # y_hat = forward_model_proj(x_hat_det.squeeze())

            seed_vector_rec = torch.zeros(y_rec.shape)
            for p_i in range(y_hat.shape[0]):  # batch size
                infect_proj = y_hat[p_i, :].cpu().clone().detach()
                seed_value = infect_proj[inf_proj_idx]
                seed_vector_rec[p_i, :][
                    np.where(
                    nodes_name_G_recived.reshape(nodes_name_G_recived.size, 1) == seed_name_received
                    )[0]] = seed_value[ np.where(
                    nodes_name_G_recived.reshape(nodes_name_G_recived.size, 1) == seed_name_received)[1]]
            x_hat_rec = seed_vector_rec.to(device)
            y_hat_rec = forward_model_rec(x_hat_rec)

            re_loss, kld, loss_vae = loss_functions.loss_vae(sf2, sf2_hat, mean, log_var)
            re_overall += re_loss.item() * x_hat.size(0)
            kld_overall += kld.item() * x_hat.size(0)

            loss_proj = loss_functions.loss_proj(y_hat, y)
            loss_rec = loss_functions.loss_rec(y_hat_rec, y_rec)


            loss_inf = loss_functions.loss_infect(y_hat, y, y_hat_rec, y_rec)
            loss = loss_vae + loss_inf
            # loss = loss_vae + loss_proj + loss_rec
            loss.backward()
            total_overall += loss.item() * x_hat.size(0)
            # loss = loss_vae + loss_proj + loss_rec
            # loss = torch.tensor([loss_vae, loss_inf])
            # loss = torch.tensor([loss_vae, loss_proj, loss_rec])
            # total_overall += (loss_vae + loss_proj + loss_rec).item() * x_hat.size(0)
            # total_overall += loss.item() * x_hat.size(0)

            # loss_vae.backward()
            # optimizer.zero_grad()
            # loss_inf.backward()
            # optimizer.zero_grad()
            # loss.backward()
            # loss.backward(gradient=torch.tensor([1.0, 1.0, 1.0]), retain_graph=True)
            # optimizer.step()

            ## Performance
            x_pred = x_hat.cpu().detach()
            x_pred[x_pred > args.seed_threshold] = 1  # 0.01
            x_pred[x_pred != 1] = 0

            seed_original = x_true.squeeze().cpu().detach().numpy().flatten()
            seed_predicted = x_pred.squeeze().cpu().detach().numpy().flatten()

            seed_accuracy_all += accuracy_score(seed_original, seed_predicted)
            seed_precision_all += precision_score(seed_original, seed_predicted, zero_division=0)
            seed_recall_all += recall_score(seed_original, seed_predicted, zero_division=0)

            y_pred = y_hat.cpu().detach()
            y_pred[y_pred > args.seed_threshold] = 1  # 0.01
            y_pred[y_pred != 1] = 0

            infected_original_proj = y.squeeze().cpu().detach().numpy().flatten()
            infected_predicted_proj = y_pred.squeeze().cpu().detach().numpy().flatten()

            accuracy_all += accuracy_score(infected_original_proj, infected_predicted_proj)
            precision_all += precision_score(infected_original_proj, infected_predicted_proj, zero_division=0)
            recall_all += recall_score(infected_original_proj, infected_predicted_proj, zero_division=0)

            y_pred_rec = y_hat_rec.cpu().detach()
            y_pred_rec[y_pred_rec > args.seed_threshold] = 1  # 0.01
            y_pred_rec[y_pred_rec != 1] = 0

            infected_original_rec = y_rec.squeeze().cpu().detach().numpy().flatten()
            infected_predicted_rec = y_pred_rec.squeeze().cpu().detach().numpy().flatten()

            accuracy_all_rec += accuracy_score(infected_original_rec.astype(int), infected_predicted_rec.astype(int))
            precision_all_rec += precision_score(infected_original_rec.astype(int), infected_predicted_rec.astype(int), zero_division=0)
            recall_all_rec += recall_score(infected_original_rec.astype(int), infected_predicted_rec.astype(int), zero_division=0)

        epoch_toc = time.perf_counter()
        epoch_time = epoch_toc-epoch_tic
        epoch_dict['reconstruction'].append(re_overall / (batch_idx + 1))
        epoch_dict['kld'].append(kld_overall / (batch_idx + 1))
        epoch_dict['total'].append(total_overall / (batch_idx + 1))
        epoch_dict['seed accuracy'].append(seed_accuracy_all / (batch_idx + 1))
        epoch_dict['seed precision'].append(seed_precision_all / (batch_idx + 1))
        epoch_dict['seed recall'].append(seed_recall_all / (batch_idx + 1))
        epoch_dict['accuracy'].append(accuracy_all / (batch_idx + 1))
        epoch_dict['precision'].append(precision_all / (batch_idx + 1))
        epoch_dict['recall'].append(recall_all / (batch_idx + 1))
        epoch_dict['accuracy_rec'].append(accuracy_all_rec / (batch_idx + 1))
        epoch_dict['precision_rec'].append(precision_all_rec / (batch_idx + 1))
        epoch_dict['recall_rec'].append(recall_all_rec / (batch_idx + 1))
        epoch_dict['time'].append(epoch_time)
        print("Epoch: {}".format(epoch + 1),
              "\tReconstruction: {:.4f}".format(re_overall / (batch_idx + 1)),
              "\tKLD: {:.4f}".format(kld_overall / (batch_idx + 1)),
              "\tTotal: {:.4f}".format(total_overall / (batch_idx + 1)),
          "\tSeed Accuracy: {:.4f}".format(seed_accuracy_all / (batch_idx + 1)),
          "\tSeed Precision: {:.4f}".format(seed_precision_all / (batch_idx + 1)),
          "\tSeed Recall: {:.4f}".format(seed_recall_all / (batch_idx + 1)),
              "\tAccuracy: {:.4f}".format(accuracy_all / (batch_idx + 1)),
              "\tPrecision: {:.4f}".format(precision_all / (batch_idx + 1)),
              "\tRecall: {:.4f}".format(recall_all / (batch_idx + 1)),
              "\tAccuracy_rec: {:.4f}".format(accuracy_all_rec / (batch_idx + 1)),
              "\tPrecision_rec: {:.4f}".format(precision_all_rec / (batch_idx + 1)),
              "\tRecall_rec: {:.4f}".format(recall_all_rec / (batch_idx + 1)),
              "\tTime Taken: {:.6f}".format(epoch_time),
              )
    # saving models
    feature_representer_chk_file = args.model_loc + 'feature_representer_final_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
        10 * args.seed_rate) + '.ckpt'
    vae_chk_file = args.model_loc + 'VAE_final_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
        10 * args.seed_rate) + '.ckpt'
    frwrd_proj_file = args.model_loc + 'forward_model_proj_final_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
        10 * args.seed_rate) + '.ckpt'
    frwrd_rec_file = args.model_loc + 'forward_model_rec_final_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
        10 * args.seed_rate) + '.ckpt'
    torch.save(feature_representer.state_dict(), feature_representer_chk_file)
    print(' Feature Representaer Model saved')

    torch.save(vae_model.state_dict(), vae_chk_file)
    print(' VAE Model saved')

    torch.save(forward_model_proj.state_dict(), frwrd_proj_file)
    print('Diffusion model for projection graph saved')

    torch.save(forward_model_rec.state_dict(), frwrd_rec_file)
    print('Diffusion model for receipient graph saved')

    epoch_df = pd.DataFrame.from_dict(epoch_dict)
    epoch_df.index = epoch_df.index.rename('epochs')
    epoch_df.to_csv(epoch_log_file)