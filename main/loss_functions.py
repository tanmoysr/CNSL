import torch
import torch.nn.functional as F

def loss_vae(sf2, sf2_hat, mean, log_var):
    beta = 1
    x, static_features, x_features = sf2
    x_hat, static_features_hat, x_features_hat = sf2_hat
    mean_sd, mean_ft, mean_sd_ft = mean
    log_var_sd, log_var_ft, log_var_sd_ft = log_var

    ### binary_cross_entropy, l1_loss, mse_loss, binary_cross_entropy_with_logits
    # reproduction_loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction='sum') # 'sum
    # reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='mean') # 'sum
    reproduction_loss_sd = F.binary_cross_entropy(x_hat, x, reduction='mean')
    # reproduction_loss_ft = F.mse_loss(static_features_hat, static_features, reduction='mean')
    # reproduction_loss_sd_ft = F.mse_loss(x_features_hat, x_features, reduction='mean')
    # reproduction_loss_sd_ft = F.binary_cross_entropy(x_hat[:,:,0].unsqueeze(1), x[:,:,0].unsqueeze(1), reduction='mean')+ \
    #                           F.mse_loss(x_features_hat[:,:,1:], x_features[:,:,1:], reduction='mean')

    reproduction_loss = reproduction_loss_sd # Must
    # reproduction_loss = reproduction_loss_sd + reproduction_loss_ft # No impact
    # reproduction_loss = reproduction_loss_sd + 1* reproduction_loss_ft + 1*reproduction_loss_sd_ft # Damaging Impact


    KLD_sd = -0.5*torch.sum(1+ log_var_sd - mean_sd.pow(2) - log_var_sd.exp())
    # KLD_ft = -0.5*torch.sum(1+ log_var_ft - mean_ft.pow(2) - log_var_ft.exp())
    # KLD_sd_ft = -0.5*torch.sum(1+ log_var_sd_ft - mean_sd_ft.pow(2) - log_var_sd_ft.exp())
    # KLD = [KLD_sd, KLD_ft, KLD_sd_ft]
    KLD = KLD_sd # Must
    # KLD = KLD_sd+KLD_ft # No impact
    # KLD = KLD_sd+KLD_ft+KLD_sd_ft # Damaging impact

    # total_loss = reproduction_loss + KLD_sd + KLD_ft + KLD_sd_ft
    total_loss = reproduction_loss + beta * KLD

    return reproduction_loss, KLD, total_loss

def loss_infect(y_hat, y, y_hat_rec, y_rec):
    # forward_loss_proj = F.mse_loss(y_hat, y, reduction='sum')
    forward_loss_proj = F.binary_cross_entropy(y_hat, y, reduction='mean')
    # forward_loss_rec = F.mse_loss(y_hat_rec, y_rec, reduction='sum')
    forward_loss_rec = F.binary_cross_entropy(y_hat_rec, y_rec, reduction='mean')
    monotone_loss_proj = torch.sum(torch.relu(y_hat-y_hat[0]))
    monotone_loss_rec = torch.sum(torch.relu(y_hat_rec-y_hat_rec[0]))
    total_loss = forward_loss_proj + forward_loss_rec + monotone_loss_proj + monotone_loss_rec
    return total_loss

def loss_proj(y_hat, y):
    forward_loss_proj = F.mse_loss(y_hat, y, reduction='mean')
    # forward_loss_proj = F.binary_cross_entropy(y_hat, y, reduction='sum')
    # monotone_loss_proj = torch.sum(torch.sigmoid(y_hat-y_hat[0]))
    # monotone_loss_proj = torch.sum(torch.relu(y_hat - y_hat[0]))
    # monotone_loss_proj = torch.sum(y_hat - y_hat[0])
    # total_loss = forward_loss_proj + monotone_loss_proj # monotone_loss is making model unstable
    total_loss = forward_loss_proj
    return total_loss

def loss_rec(y_hat, y):
    # forward_loss_rec = F.mse_loss(y_hat, y, reduction='mean')
    forward_loss_rec = F.binary_cross_entropy(y_hat, y, reduction='sum')
    # monotone_loss_rec = torch.sum(torch.sigmoid(y_hat-y_hat[0]))
    # monotone_loss_rec = torch.sum(torch.relu(y_hat - y_hat[0])) # Unstable
    # monotone_loss_rec = torch.sum(y_hat - y_hat[0])
    # total_loss = forward_loss_rec + monotone_loss_rec # monotone_loss is making model unstable
    total_loss = forward_loss_rec
    return total_loss


def loss_inverse_initial(y, y_hat, y_rec, y_hat_rec, x_hat, f_z):
    forward_loss_proj = F.mse_loss(y_hat, y)
    forward_loss_rec = F.mse_loss(y_hat_rec, y_rec)
    forward_loss = forward_loss_proj + forward_loss_rec

    pdf_sum = 0

    for i, x_i in enumerate(x_hat[0]):
        temp = torch.pow(f_z[i], x_i) * torch.pow(1 - f_z[i], 1 - x_i).to(torch.double)
        pdf_sum += torch.log(temp)

    return forward_loss - pdf_sum, pdf_sum


def loss_inverse(y, y_hat, y_rec, y_hat_rec, x_hat, f_z_all, BN, device):
    forward_loss_proj = F.mse_loss(y_hat, y)
    forward_loss_rec = F.mse_loss(y_hat_rec, y_rec)
    forward_loss = forward_loss_proj + forward_loss_rec

    log_pmf = []
    for f_z in f_z_all:
        log_likelihood_sum = torch.zeros(1).to(device)
        for i, x_i in enumerate(x_hat[0]):
            temp = torch.pow(f_z[i], x_i) * torch.pow(1 - f_z[i], 1 - x_i).to(torch.double)
            log_likelihood_sum += torch.log(temp)
        log_pmf.append(log_likelihood_sum)

    log_pmf = torch.stack(log_pmf)
    log_pmf = BN(log_pmf.float())

    pmf_max = torch.max(log_pmf)

    pdf_sum = pmf_max + torch.logsumexp(log_pmf - pmf_max, dim=0)

    return forward_loss - pdf_sum, forward_loss