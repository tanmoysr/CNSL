import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from datetime import datetime
from main.configuration import args
# To check all the parameters
# print(plt.rcParams)

# get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# convert datetime obj to string
str_current_datetime = str(current_datetime)

# epoch_log_file = args.model_loc + 'train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
#     10 * args.seed_rate) + '.csv'
# epoch_log_file_vae = args.model_loc + 'VAE_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
#     10 * args.seed_rate) + '.csv'
# epoch_log_file_diffProj = args.model_loc + 'diffProj_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
#     10 * args.seed_rate) + '.csv'
# epoch_log_file_diffRec = args.model_loc + 'diffRec_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
#     10 * args.seed_rate) + '.csv'
epoch_log_file_infer = args.model_loc + 'inference_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
fig_file = args.model_loc + 'Inference_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + str_current_datetime+ '.png'

# epoch_df = pd. read_csv(epoch_log_file)
# epoch_df_vae = pd. read_csv(epoch_log_file_vae)
# epoch_df_diffProj = pd. read_csv(epoch_log_file_diffProj)
# epoch_df_diffRec = pd. read_csv(epoch_log_file_diffRec)
epoch_df_infer = pd. read_csv(epoch_log_file_infer)

# Reseting plot parameter to default
plt.rcdefaults()
# Updating Parameters for Paper
params = {
    'lines.linewidth' : 2,
    'lines.markersize' : 12,
   'axes.labelsize': 12,
    'axes.titlesize':12,
    'axes.titleweight':'bold',
    'font.size': 8,
    'font.family': 'Arial', # 'Times New Roman'
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
    'figure.autolayout': True,
   'figure.figsize': [12, 6] # width and height in inch (max width = 7.5", max length = 10")
   }
plt.rcParams.update(params)
fig, ax = plt.subplots(4,3)

# ax[0,0].plot(epoch_df['epochs'], epoch_df['reconstruction'], label="reconstruction")
ax[0,0].plot(epoch_df_infer['epochs'], epoch_df_infer['forward_loss'], label="forward loss")
ax[0,0].set(xlabel='epochs', ylabel='loss',title='forward loss')
ax[0,0].set_yscale("log")
# ax[0,0].legend(loc="upper right")
# ax[0,0].grid()

# ax[0,1].plot(epoch_df['epochs'], epoch_df['kld'], label="kld")
ax[0,1].plot(epoch_df_infer['epochs'], epoch_df_infer['loss'], label="loss")
ax[0,1].set(xlabel='epochs', ylabel='KLD',title='loss')
# ax[0,1].set_yscale("log")

# ax[0,2].plot(epoch_df['epochs'], epoch_df['total'], label="total")
ax[0,2].plot(epoch_df_infer['epochs'], epoch_df_infer['loss'], label="total")
ax[0,2].set(xlabel='epochs', ylabel='total',title='total')
# ax[0,2].set_yscale("log")

# ax[1,0].plot(epoch_df['epochs'], epoch_df['precision'], label="recprecisiononstruction")
ax[1,0].plot(epoch_df_infer['epochs'], epoch_df_infer['seed precision'], label="precision_seed")
ax[1,0].set(xlabel='epochs', ylabel='precision',title='seed precision '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[1,0].set_yscale("log")
# ax[1,0].legend(loc="upper right")
# ax[1,0].grid()

ax[1,1].plot(epoch_df_infer['epochs'], epoch_df_infer['seed recall'], label="recall_seed")
ax[1,1].set(xlabel='epochs', ylabel='recall',title='recall '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[1,1].set_yscale("log")

ax[1,2].plot(epoch_df_infer['epochs'], epoch_df_infer['seed accuracy'], label="accuracy_seed")
ax[1,2].set(xlabel='epochs', ylabel='accuracy',title='seed accuracy '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[1,2].set_yscale("log")
# ax[1,2].legend(loc="upper right")
# ax[1,2].grid()

# ax[1,2].plot(epoch_df['epochs'], epoch_df['time'], label="time")
# ax[1,2].set(xlabel='epochs', ylabel='time',title='time')
# # ax[1,2].set_yscale("log")

ax[2,0].plot(epoch_df_infer['epochs'], epoch_df_infer['precision'], label="precesion_project")
ax[2,0].set(xlabel='epochs', ylabel='precision',title='projecting precision '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[2,0].set_yscale("log")
# ax[2,0].legend(loc="upper right")
# ax[2,0].grid()

ax[2,1].plot(epoch_df_infer['epochs'], epoch_df_infer['recall'], label="recall_project")
ax[2,1].set(xlabel='epochs', ylabel='recall',title='projecting recall '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[2,1].set_yscale("log")

ax[2,2].plot(epoch_df_infer['epochs'], epoch_df_infer['accuracy'], label="accuracy_project")
ax[2,2].set(xlabel='epochs', ylabel='accuracy',title='projecting accuracy '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[2,2].set_yscale("log")
# ax[2,2].legend(loc="upper right")
# ax[2,2].grid()

ax[3,0].plot(epoch_df_infer['epochs'], epoch_df_infer['precision_rec'], label="precision_receipt")
ax[3,0].set(xlabel='epochs', ylabel='precision',title='receiving precision '+args.dataset.split("_")[0].split("2")[1] + ' ' + args.diffusion_model_rec)
# ax[2,0].set_yscale("log")
# ax[2,0].legend(loc="upper right")
# ax[2,0].grid()

ax[3,1].plot(epoch_df_infer['epochs'], epoch_df_infer['recall_rec'], label="recall_receipt")
ax[3,1].set(xlabel='epochs', ylabel='recall',title='receiving recall '+args.dataset.split("_")[0].split("2")[1] + ' ' + args.diffusion_model_rec)
# ax[2,1].set_yscale("log")

ax[3,2].plot(epoch_df_infer['epochs'], epoch_df_infer['accuracy_rec'], label="accuracy_receipt")
ax[3,2].set(xlabel='epochs', ylabel='accuracy',title='receiving accuracy '+args.dataset.split("_")[0].split("2")[1] + ' ' + args.diffusion_model_rec)
# ax[2,2].set_yscale("log")
# ax[2,2].legend(loc="upper right")
# ax[2,2].grid()

fig.suptitle("Inference")
# fig.savefig("../saved_models/VAE_train_github2stack_LT10"+str_current_datetime+".png")
fig.savefig(fig_file)

plt.show()