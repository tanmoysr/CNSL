# from configuration import args
import argparse
import json

parser = argparse.ArgumentParser(description="CrossNet")
#%% data setup
datasets = ['github2stack_', 'colocation2social_']
parser.add_argument("-d", "--dataset", default="colocation2social_", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
diffusion = ['IC', 'LT', 'SIS', 'SimA', 'SimB']
parser.add_argument("-dm_p", "--diffusion_model_proj", required=False, default="SimA", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
parser.add_argument("-dm_r", "--diffusion_model_rec", required=False, default="SimA", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))
seed_rate = [0, 1] # In dataset 0, 10
parser.add_argument("-sp", "--seed_rate", required=False, default=0, type=int,
                    help="one of: {}".format(", ".join(str(sorted(seed_rate)))))
sample = [100, 98, 85] # In dataset 100 for github, 98 for SimA & 85 for SimB
parser.add_argument("-sam", "--sample", required=False, default=98, type=int,
                    help="Data samples from one of: {}".format(", ".join(str(sorted(sample)))))
input_features = [4, 15] # Github: 4, Social: 15
parser.add_argument("-in_f_dim", "--input_feature_dim", required=False, default=15, type=int,
                    help="one of: {}".format(", ".join(str(sorted(input_features)))))
parser.add_argument("-dl", "--data_loc", required=False, default='../data/', type=str,
                    help="Data samples")
#%% Model Setup
parser.add_argument("-ml", "--model_loc", required=False, default='../saved_models/', type=str,
                    help="Data samples")
# parser.add_argument("-splitTrain", "--split_train", required=False, default=[False,False,True], type=bool,
#                     help="[VAE, Diff1, Diff2]")
parser.add_argument("-e", "--numEpoch", required=False, default=15, type=int,
                    help="Number of Epochs") # 3 for Git2Stack; 15 for Sim,
parser.add_argument("-bs", "--batchSize", required=False, default=2, type=int,
                    help="Batch Size. Minimum=2") # 2
parser.add_argument("-bsInfer", "--batchSizeInfer", required=False, default=2, type=int,
                    help="Batch Size. Minimum=2") # 2
parser.add_argument("-st", "--seed_threshold", required=False, default=0.31, type=float,
                    help="Seed threshold for Prediction") # 0.31 for simulation
#%% VAE setup
parser.add_argument("-eVAE", "--numEpochVAE", required=False, default=30, type=int,
                    help="Number of Epochs for VAE") # 30 for All,
parser.add_argument("-lrVAE", "--lr_VAE", required=False, default=1e-4, type=float,
                    help="Learning rate of VAE") # 1e-4
parser.add_argument("-wdVAE", "--wd_VAE", required=False, default=1e-5, type=float,
                    help="Weight decay for VAE") # 1e-5
#%% Diffusion projection setup
parser.add_argument("-eDiffProj", "--numEpochDiffProj", required=False, default=15, type=int,
                    help="Number of Epochs for Diffusion Projection") # 15 for All
parser.add_argument("-lrDiffProj", "--lr_DiffProj", required=False, default=5e-3, type=float,
                    help="Learning rate of Diffusion Projection") # 5e-3
parser.add_argument("-wdDiffProj", "--wd_DiffProj", required=False, default=0, type=float,
                    help="Weight decay for Diffusion Projection")
#%% Diffusion receiving setup
parser.add_argument("-proj2recMap", "--select_proj2recMap", required=False, default=True, type=bool,
                    help="Mapping projection infected nodes to receiving seed nodes")
parser.add_argument("-eDiffRec", "--numEpochDiffRec", required=False, default=10, type=int,
                    help="Number of Epochs for Diffusion Receiving") # 5 for Git2Stack; 10 for Sim
parser.add_argument("-lrDiffRec", "--lr_DiffRec", required=False, default=1e-2, type=float,
                    help="Learning rate of Diffusion Receiving")
parser.add_argument("-wdDiffRec", "--wd_DiffRec", required=False, default=0, type=float,
                    help="Weight decay for Diffusion Receiving")
#%% Inference setup
parser.add_argument("-eInfer", "--numEpochInfer", required=False, default=3, type=int,
                    help="Number of Epochs for Inference") # 3 for All
parser.add_argument("-lrInfer", "--lr_Infer", required=False, default=1e-10, type=float,
                    help="Learning rate of Inference")
parser.add_argument("-wdInfer", "--wd_Infer", required=False, default=1e-1, type=float,
                    help="Weight decay for Inference")
#%% Rest of the setup
args = parser.parse_args(args=[]) # use for jupyter notebook
# args = parser.parse_args()
with open('../saved_models/Params_train_' + args.dataset + '_' + args.diffusion_model_proj + '2' + args.diffusion_model_rec  + str(10 * args.seed_rate) + '.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)