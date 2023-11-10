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


epoch_log_file_diffProj = args.model_loc + 'diffProj_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + '.csv'
fig_file_diffProj = args.model_loc + 'diffProj_train_log_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec + str(
    10 * args.seed_rate) + str_current_datetime+ '.png'

# epoch_df = pd. read_csv(epoch_log_file)
epoch_df_diffProj = pd. read_csv(epoch_log_file_diffProj)
# epoch_df_diffProj = pd. read_csv(epoch_log_file_diffProj)
# epoch_df_diffRec = pd. read_csv(epoch_log_file_diffRec)

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
   'figure.figsize': [12, 3] # width and height in inch (max width = 7.5", max length = 10")
   }
plt.rcParams.update(params)
fig, ax = plt.subplots(1,4)

# ax[0,2].plot(epoch_df['epochs'], epoch_df['total'], label="total")
ax[0].plot(epoch_df_diffProj['epochs'], epoch_df_diffProj['total_diffProj'], label="total")
ax[0].set(xlabel='epochs', ylabel='total',title='total')
# ax[0,2].set_yscale("log")

# ax[1,0].plot(epoch_df['epochs'], epoch_df['precision'], label="recprecisiononstruction")
ax[1].plot(epoch_df_diffProj['epochs'], epoch_df_diffProj['infect_precision'], label="precision_diffProj")
ax[1].set(xlabel='epochs', ylabel='precision',title='precision '+args.dataset.split("_")[0].split("2")[0])
# ax[1,0].set_yscale("log")
# ax[1,0].legend(loc="upper right")
# ax[1,0].grid()

ax[2].plot(epoch_df_diffProj['epochs'], epoch_df_diffProj['infect_recall'], label="recall_diffProj")
ax[2].set(xlabel='epochs', ylabel='recall',title='recall '+args.dataset.split("_")[0].split("2")[0])
# ax[1,1].set_yscale("log")

ax[3].plot(epoch_df_diffProj['epochs'], epoch_df_diffProj['infect_accuracy'], label="accuracy_diffProj")
ax[3].set(xlabel='epochs', ylabel='accuracy',title='accuracy '+args.dataset.split("_")[0].split("2")[0])
# ax[1,2].set_yscale("log")
# ax[1,2].legend(loc="upper right")
# ax[1,2].grid()
# fig.suptitle("Diff Proj Training Log")

fig.suptitle("Training Projecting Network Diffusion Model")

fig.savefig(fig_file_diffProj)

plt.show()