import json
import numpy as np
import networkx as nx
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch
import pandas as pd
import pickle

def json2graph(feature_file, edge_file):
    with open(feature_file) as json_file:
        static_features = json.load(json_file)
    G = nx.Graph()
    G.add_nodes_from([(node, {**attr}) for (node, attr) in static_features.items()])
    df = pd.read_csv(edge_file, delim_whitespace=True, header=None)
    for index, row in df.iterrows():
        G.add_edge(row[0], row[1])
    node_name_list = []
    static_features = []
    delete_node = []
    for node_name, feature_dict in list(G.nodes(data=True)):
        if len(list(feature_dict.values())[1:-1]) == 0:
            delete_node.append(node_name)
        else:
            node_name_list.append(node_name)
            # static_features.append(list(feature_dict.values())[1:-1])
            static_features.append(list(feature_dict.values())[2:-1])
    G.remove_nodes_from(delete_node)
    adj_matrix = nx.to_numpy_array(G, dtype='f')
    static_array = np.asarray(static_features, dtype=np.float32)

    return G, adj_matrix, static_array

def generate_seed_vector(top_nodes, seed_num, G):
    seed_nodes = random.sample(top_nodes, seed_num)
    seed_vector = [1 if node in seed_nodes else 0 for node in G.nodes()]
    return seed_vector

def infected_nodes(G, seed_vector_init, inf_vec_all, diffusion='LT', diff_num = 1, iter_num = 100):
    seed_vector = [i for i in range(len(seed_vector_init)) if seed_vector_init[i] == 1]
    # print(seed_vector)
    for j in range(diff_num):
        if diffusion == 'LT':
            model = ep.ThresholdModel(G)
            config = mc.Configuration()
            for n in G.nodes():
                config.add_node_configuration("threshold", n, 0.5)
        elif diffusion == 'IC':
            model = ep.IndependentCascadesModel(G)
            config = mc.Configuration()
            for e in G.edges():
                config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]]) # Play with it
        elif diffusion == 'SIS':
            model = ep.SISModel(G)
            config = mc.Configuration()
            config.add_model_parameter('beta', 0.001)
            config.add_model_parameter('lambda', 0.001)
        else:
            raise ValueError('Only IC, LT and SIS are supported.')

        config.add_model_initial_configuration("Infected", seed_vector)

        model.set_initial_status(config)

        iterations = model.iteration_bunch(iter_num)

        node_status = iterations[0]['status']

        for k in range(1, len(iterations)):
            node_status.update(iterations[k]['status'])

        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1

        inf_vec_all += inf_vec
    return inf_vec_all/diff_num

def cross_data_generation(G_proj_org, adj_proj, static_proj, G_received_org, adj_received, proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='LT', diffusion_recived='IC', dataset='github2stack'):

    nodes_name_G_proj = np.array(list(G_proj_org.nodes()))
    nodes_name_G_recived = np.array(list(G_received_org.nodes()))
    df_proj2recived = pd.read_csv(proj2recived_file, delim_whitespace=True, header=None)
    proj_nodes = df_proj2recived[0].to_numpy() #0
    receipient_nodes = df_proj2recived[1].to_numpy()#1

    G_proj = nx.from_numpy_array(adj_proj)
    node_num_proj = len(G_proj.nodes())
    seed_num_proj = int(percentage * node_num_proj / 100)
    samples_proj = []

    degree_list_proj = list(G_proj.degree())
    degree_list_proj.sort(key=lambda x: x[1], reverse=True)
    top_nodes_proj = [x[0] for x in degree_list_proj[:int(len(degree_list_proj) * 0.3)]] # shrinking the seed 0.3->0.1. Proportional to network.
    ''''
    Network size is 1000
    We want to predict 50
    That time 0.3->0.05
    100 samples
    80 for training
    20 for inference
    '''

    G_received = nx.from_numpy_array(adj_received)
    node_num_received = len(G_received.nodes())
    # seed_num_received = int(percentage * node_num_received / 100)
    samples_received = []

    for j in range(nums):
        print('Sample {} generating'.format(j))
        seed_vector_proj = generate_seed_vector(top_nodes_proj, seed_num_proj, G_proj)
        inf_vec_all_proj = torch.zeros(node_num_proj)
        inf_vec_all_proj = infected_nodes(G_proj, seed_vector_proj, inf_vec_all_proj, diffusion=diffusion_proj,
                                          diff_num=1, iter_num=100)
        samples_proj.append([seed_vector_proj, inf_vec_all_proj])

        inf_proj_idx = []
        for i in nodes_name_G_proj[inf_vec_all_proj == 1]:
            inf_proj_idx.extend(np.where(proj_nodes == i)[0].tolist())
        # seed_name_received = nodes_name_G_recived[inf_proj_idx]
        seed_name_received = receipient_nodes[inf_proj_idx]
        seed_vector_received = []
        for index, element in enumerate(nodes_name_G_recived):
            seed_vector_received.append(1) if str(element) in seed_name_received else seed_vector_received.append(0)
        inf_vec_all_received = torch.zeros(node_num_received)
        inf_vec_all_received = infected_nodes(G_received, seed_vector_received, inf_vec_all_received,
                                              diffusion=diffusion_recived, diff_num=1, iter_num=100)
        samples_received.append([seed_vector_received, inf_vec_all_received])
    # Changing shape from [samples, 2, nodes] to [samples, nodes, 2]
    samples_proj = torch.Tensor(samples_proj).permute(0, 2, 1)
    samples_received = torch.Tensor(samples_received).permute(0, 2, 1)

    data_dict = {'original_graph_proj': G_proj_org,
                 'adj_proj': adj_proj,
                 'prob_proj': adj_proj,
                 'inverse_pairs_proj': samples_proj,
                 'static_features_proj': static_proj,
                 'proj_nodes': proj_nodes,
                 'original_graph_received': G_received_org,
                 'adj_received': adj_received,
                 'prob_received': adj_received,
                 'inverse_pairs_received': samples_received,
                 'receipient_nodes': receipient_nodes}

    f = open('{}_{}2{}_{}_{}.SG'.format(dataset, diffusion_proj, diffusion_recived, percentage, nums), 'wb')
    pickle.dump(data_dict, f)
    f.close()
    print('Data generation finished')

if __name__ == '__main__':
    root_directory = "../data/communication_net_raw/"

    # Projection network:
    features_file_git = root_directory+'static_features_git.json'
    edges_file_git = root_directory+'edges_git.txt'
    G_git, adj_matrix_git, static_array_git = json2graph(features_file_git, edges_file_git)

    # Receiving network
    with open(root_directory+'G_stack.pkl', 'rb') as fp:
        G_stack = pickle.load(fp)
    adj_matrix_stack = nx.to_numpy_array(G_stack, dtype='f')

    # Linking bridge
    proj2recived_file = root_directory+'gid_sid.txt'

    # Data generataion
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='LT', diffusion_recived='LT',
                          dataset='../data/github2stack')
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='LT', diffusion_recived='IC',
                          dataset='../data/github2stack')
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='LT', diffusion_recived='SIS',
                          dataset='../data/github2stack')
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='IC', diffusion_recived='LT',
                          dataset='../data/github2stack')
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='IC', diffusion_recived='IC',
                          dataset='../data/github2stack')
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='IC', diffusion_recived='SIS',
                          dataset='../data/github2stack')


