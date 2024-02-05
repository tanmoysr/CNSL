import torch
import torch.nn.functional as F

def loss_vae(sf2, sf2_hat, mean, log_var):
    beta = 1
    x, static_features, x_features = sf2
    x_hat, static_features_hat, x_features_hat = sf2_hat
    mean_sd, mean_ft, mean_sd_ft = mean
    log_var_sd, log_var_ft, log_var_sd_ft = log_var
    reproduction_loss_sd = F.binary_cross_entropy(x_hat, x, reduction='mean')
    reproduction_loss = reproduction_loss_sd 
    KLD_sd = -0.5*torch.sum(1+ log_var_sd - mean_sd.pow(2) - log_var_sd.exp())
    KLD = KLD_sd
    total_loss = reproduction_loss + beta * KLD

    return reproduction_loss, KLD, total_loss

def loss_infect(y_hat, y, y_hat_rec, y_rec):
    forward_loss_proj = F.binary_cross_entropy(y_hat, y, reduction='mean')
    forward_loss_rec = F.binary_cross_entropy(y_hat_rec, y_rec, reduction='mean')
    monotone_loss_proj = torch.sum(torch.relu(y_hat-y_hat[0]))
    monotone_loss_rec = torch.sum(torch.relu(y_hat_rec-y_hat_rec[0]))
    total_loss = forward_loss_proj + forward_loss_rec + monotone_loss_proj + monotone_loss_rec
    return total_loss

def loss_proj(y_hat, y):
    forward_loss_proj = F.binary_cross_entropy(y_hat, y, reduction='sum')
    total_loss = forward_loss_proj
    return total_loss

def loss_rec(y_hat, y):
    forward_loss_rec = F.binary_cross_entropy(y_hat, y, reduction='sum')
    total_loss = forward_loss_rec
    return total_loss

def loss_inverse(y_rec, y_hat_rec, x_hat):
    forward_loss = F.mse_loss(y_hat_rec, y_rec)
    L0_loss = torch.sum(torch.abs(x_hat))/x_hat.shape[1]
    return (forward_loss + L0_loss), L0_loss
