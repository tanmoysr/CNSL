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

epoch_log_file = args.model_loc + 'train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
epoch_log_file_vae = args.model_loc + 'VAE_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
epoch_log_file_diffProj = args.model_loc + 'diffProj_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
epoch_log_file_diffRec = args.model_loc + 'diffRec_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
fig_file = args.model_loc + 'train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + str_current_datetime+ '.png'

epoch_df = pd. read_csv(epoch_log_file)
epoch_df_vae = pd. read_csv(epoch_log_file_vae)
epoch_df_diffProj = pd. read_csv(epoch_log_file_diffProj)
epoch_df_diffRec = pd. read_csv(epoch_log_file_diffRec)

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
fig, ax = plt.subplots(3,3)

# ax[0,0].plot(epoch_df['epochs'], epoch_df['reconstruction'], label="reconstruction")
ax[0,0].plot(epoch_df['epochs'], epoch_df['seed precision'], label="reconstruction")
ax[0,0].set(xlabel='epochs', ylabel='precision',title='seed precision')
ax[0,0].set_yscale("log")
# ax[0,0].legend(loc="upper right")
# ax[0,0].grid()

# ax[0,1].plot(epoch_df['epochs'], epoch_df['kld'], label="kld")
ax[0,1].plot(epoch_df['epochs'], epoch_df['seed recall'], label="kld")
ax[0,1].set(xlabel='epochs', ylabel='recall',title='seed recall')
# ax[0,1].set_yscale("log")

# ax[0,2].plot(epoch_df['epochs'], epoch_df['total'], label="total")
ax[0,2].plot(epoch_df['epochs'], epoch_df['seed accuracy'], label="total")
ax[0,2].set(xlabel='epochs', ylabel='accuracy',title='seed accuracy')
# ax[0,2].set_yscale("log")

# ax[1,0].plot(epoch_df['epochs'], epoch_df['precision'], label="recprecisiononstruction")
ax[1,0].plot(epoch_df['epochs'], epoch_df['precision'], label="precision_vae")
ax[1,0].set(xlabel='epochs', ylabel='precision',title='precision '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[1,0].set_yscale("log")
# ax[1,0].legend(loc="upper right")
# ax[1,0].grid()

ax[1,1].plot(epoch_df['epochs'], epoch_df['recall'], label="recall_vae")
ax[1,1].set(xlabel='epochs', ylabel='recall',title='recall '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[1,1].set_yscale("log")

ax[1,2].plot(epoch_df['epochs'], epoch_df['accuracy'], label="accuracy_vae")
ax[1,2].set(xlabel='epochs', ylabel='accuracy',title='accuracy '+args.dataset.split("_")[0].split("2")[0] + ' ' + args.diffusion_model_proj)
# ax[1,2].set_yscale("log")
# ax[1,2].legend(loc="upper right")
# ax[1,2].grid()

# ax[1,2].plot(epoch_df['epochs'], epoch_df['time'], label="time")
# ax[1,2].set(xlabel='epochs', ylabel='time',title='time')
# # ax[1,2].set_yscale("log")

ax[2,0].plot(epoch_df['epochs'], epoch_df['precision_rec'], label="recprecisiononstruction")
ax[2,0].set(xlabel='epochs', ylabel='precision',title='precision '+args.dataset.split("_")[0].split("2")[1] + ' ' + args.diffusion_model_rec)
# ax[2,0].set_yscale("log")
# ax[2,0].legend(loc="upper right")
# ax[2,0].grid()

ax[2,1].plot(epoch_df['epochs'], epoch_df['recall_rec'], label="recall")
ax[2,1].set(xlabel='epochs', ylabel='recall',title='recall '+args.dataset.split("_")[0].split("2")[1] + ' ' + args.diffusion_model_rec)
# ax[2,1].set_yscale("log")

ax[2,2].plot(epoch_df['epochs'], epoch_df['accuracy_rec'], label="recprecisiononstruction")
ax[2,2].set(xlabel='epochs', ylabel='accuracy',title='accuracy '+args.dataset.split("_")[0].split("2")[1] + ' ' + args.diffusion_model_rec)
# ax[2,2].set_yscale("log")
# ax[2,2].legend(loc="upper right")
# ax[2,2].grid()

fig.suptitle("Training All Models")

# fig.savefig("../saved_models/VAE_train_github2stack_LT10"+str_current_datetime+".png")
fig.savefig(fig_file)

plt.show()