# Importing Libraries
import numpy as np
import pickle
import pandas as pd
import networkx as nx
import torch

def text2graph(node_file, edge_file, weight = False):
    # Generate graph from observed edges
    if node_file !=None:
        nodes_list = pd.read_csv(node_file, delim_whitespace=True, header=None).astype(str)[0].values.tolist()
        G = nx.Graph()
        G.add_nodes_from(nodes_list)
    else:
        G = nx.Graph()
    df = pd.read_csv(edge_file, delim_whitespace=True, header=None).astype(str)
    for index, row in df.iterrows():
        if weight:
            G.add_edge(row[0], row[1], weight=row[2])
        else:
            G.add_edge(row[0], row[1])
    adj_matrix = nx.to_numpy_array(G, dtype='f')
    return G, adj_matrix

def infoProp(misinformation_file):
    # read and check misinformation
    with open(misinformation_file, "r") as inputFile:
        misinformation_lines = inputFile.readlines()
    misinformation_dict ={'Sample':[],
                          'Day':[],
                         'From': [],
                         'To': [],
                          'Direction': []}
    seed_dict = {'Sample':[],
                 'SeedSet':[]}
    sample = 0
    for line in misinformation_lines:
        if 'Day' in line:
            day = line.strip().split(' ')[1][1:]
        elif 'New Spread!' in line:
            sample+=1
            list_of_seed = line.strip().split(': ')[1].split(' ')
            seed_dict['Sample'].append(sample)
            seed_dict['SeedSet'].append(list_of_seed)
        else:
            # print(line)
            try:
                poject, receive, direction = line.strip().split(' ')
                misinformation_dict['Sample'].append(sample)
                misinformation_dict['Day'].append(day)
                misinformation_dict['From'].append(poject)
                misinformation_dict['To'].append(receive)
                misinformation_dict['Direction'].append(direction)
            except:
                print(line)
    misinformation_df = pd.DataFrame.from_dict(misinformation_dict)
    misinformation_df.index.names = ['Count']
    seed_df = pd.DataFrame.from_dict(seed_dict)
    seed_df.index.names = ['Count']
    # create unique list of samples
    UniqueSample = misinformation_df.Sample.unique()

    # create a data frame dictionary to store your data frames
    DataFrameDict = {elem: pd.DataFrame() for elem in UniqueSample}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = misinformation_df[:][misinformation_df.Sample == key]
    return DataFrameDict, seed_df

def infoProp2(misinformation_file):
    # read and check misinformation
    with open(misinformation_file, "r") as inputFile:
        misinformation_lines = inputFile.readlines()
    misinformation_dict ={'Sample':[],
                          'Day':[],
                         'From': [],
                         'To': [],
                          'Direction': []}
    seed_dict = {'Sample':[],
                 'SeedSet':[]}
    sample = 0
    seed_increase = False
    for line in misinformation_lines:
        if 'Day' in line:
            day = line.strip().split(' ')[1][1:]
            seed_increase = False
        elif 'New Spread!' in line:
            sample+=1
            list_of_seed = line.strip().split(': ')[1].split(' ')
            seed_dict['Sample'].append(sample)
            seed_dict['SeedSet'].append(list_of_seed)
            seed_increase = True
        else:
            # print(line)
            try:
                poject, receive, direction = line.strip().split(' ')
                misinformation_dict['Sample'].append(sample)
                misinformation_dict['Day'].append(day)
                misinformation_dict['From'].append(poject)
                misinformation_dict['To'].append(receive)
                misinformation_dict['Direction'].append(direction)
                if seed_increase:
                    seed_dict['SeedSet'][-1].append(poject)
                    seed_dict['SeedSet'][-1].append(receive)
            except:
                print(line)
    misinformation_df = pd.DataFrame.from_dict(misinformation_dict)
    misinformation_df.index.names = ['Count']
    seed_df = pd.DataFrame.from_dict(seed_dict)
    seed_df.index.names = ['Count']
    # create unique list of samples
    UniqueSample = misinformation_df.Sample.unique()

    # create a data frame dictionary to store your data frames
    DataFrameDict = {elem: pd.DataFrame() for elem in UniqueSample}

    for key in DataFrameDict.keys():
        DataFrameDict[key] = misinformation_df[:][misinformation_df.Sample == key]
    return DataFrameDict, seed_df

def all_features(feature_file):
    # Feature processing
    # Read and check the features
    with open(feature_file) as inputFile:
        c2s_features_df = pd.read_csv(inputFile, sep=" ")
        c2s_features_df['agent_id'] = c2s_features_df['agent_id'].astype(str)
    one_hot = pd.get_dummies(c2s_features_df['agent_interest'])
    # Drop column agent_interest as it is now encoded
    c2s_features_df = c2s_features_df.drop('agent_interest', axis=1)
    # Join the encoded df
    c2s_features_df = c2s_features_df.join(one_hot)
    all_ft_dict = c2s_features_df.set_index('agent_id').T.to_dict('dict')
    return all_ft_dict

def sim_data_generation(root_directory, net_t, dataset, initial_seed_only = True, crop_limit = 98):
    print("Net type: {}".format(net_t))
    # Folder names
    data_folder = '{}{}/'.format(root_directory, net_t)
    # File names
    misinformation_file = '{}misinformation_edges.txt'.format(data_folder)
    social_file = '{}observed_social_network_edges.txt'.format(data_folder)
    colocation_file = '{}observed_colocation_network_edges.txt'.format(data_folder)
    feature_file = '{}features.txt'.format(data_folder)
    print(data_folder, feature_file, misinformation_file, colocation_file, social_file)

    # %% Feature processing
    all_ft_dict = all_features(feature_file)  # 4000 agents

    # %% Misinformation Processing:
    if initial_seed_only:
        DataFrameDict, seed_df = infoProp(misinformation_file)
    else:
        DataFrameDict, seed_df = infoProp2(misinformation_file)

    # %% Graphs processing
    # Generate graph from observed edges of social
    G_social, adj_matrix_social = text2graph(None, social_file, False)
    print("Social Nodes {}, Edges {}".format(len(list(G_social.nodes())), len(list(G_social.edges()))))
    # Generate graph from observed edges of colocation
    G_colocation, adj_matrix_colocation = text2graph(None, colocation_file, False)
    print("Colocation Nodes {}, Edges {}".format(len(list(G_colocation.nodes())), len(list(G_colocation.edges()))))

    for sample_key, misinformation_df in DataFrameDict.items():
        # Adding any missing edges to social net
        social_misinfo = misinformation_df.loc[misinformation_df['Direction'] == '2'][['From', 'To']]
        for index, row in social_misinfo.iterrows():
            try:
                if row[1] not in G_social.neighbors(row[0]):
                    G_social.add_edge(row[0], row[1])
            except:
                G_social.add_edge(row[0], row[1])
        # Adding any missing edges to colocation
        colocation_misinfo = misinformation_df.loc[misinformation_df['Direction'] == '1'][['From', 'To']]
        for index, row in colocation_misinfo.iterrows():
            try:
                if row[1] not in G_colocation.neighbors(row[0]):
                    G_colocation.add_edge(row[0], row[1])
            except:
                G_colocation.add_edge(row[0], row[1])
    print("With misinformation, Social Nodes {}, Edges {}".format(len(list(G_social.nodes())),
                                                                  len(list(G_social.edges()))))
    print("With misinformation, Colocation Nodes {}, Edges {}".format(len(list(G_colocation.nodes())),
                                                                      len(list(G_colocation.edges()))))
    # %% Final graph
    selected_ft_dict = {k: v for k, v in all_ft_dict.items() if k in list(G_colocation.nodes())}
    df_temp = nx.to_pandas_edgelist(G_colocation)
    G_colocation_final = nx.Graph()
    G_colocation_final.add_nodes_from([(node, {**attr}) for (node, attr) in selected_ft_dict.items()])
    for index, row in df_temp.iterrows():
        G_colocation_final.add_edge(row[0], row[1])
    adj_matrix_col_final = nx.to_numpy_array(G_colocation_final, dtype='f')

    static_features = []
    delete_node = []
    for node_name, feature_dict in list(G_colocation_final.nodes(data=True)):
        if len(list(feature_dict.values())[1:-1]) == 0:
            delete_node.append(node_name)
        else:
            static_features.append(list(feature_dict.values()))
    G_colocation_final.remove_nodes_from(delete_node)
    adj_matrix_col_final = nx.to_numpy_array(G_colocation_final, dtype='f')
    static_array_social = np.asarray(static_features, dtype=np.float32)

    adj_matrix_social_final = nx.to_numpy_array(G_social, dtype='f')
    # %% Mapping
    colocation_nodes = list(G_colocation_final.nodes())
    social_nodes = list(G_social.nodes())
    proj2recived = {
        'social_proj_nd': [],
        'colocation_rec_nd': []
    }
    for i in colocation_nodes:
        if i in social_nodes:
            proj2recived['social_proj_nd'].append(i)
            proj2recived['colocation_rec_nd'].append(i)
    print("Mapping finished")

    # %% Portion of model data
    G_proj_org = G_colocation_final
    adj_proj = adj_matrix_col_final
    static_proj = static_array_social
    G_received_org = G_social
    adj_received = adj_matrix_social_final

    proj_nodes = np.array(proj2recived['colocation_rec_nd'])  # 0
    receipient_nodes = np.array(proj2recived['social_proj_nd'])  # 1

    # %% Samples
    samples_proj = []
    samples_rec = []
    for sample_key, misinformation_df in DataFrameDict.items():
        if sample_key>crop_limit:
            sample_key -= 1
            break
        print(sample_key)
        social_misinfo = misinformation_df.loc[misinformation_df['Direction'] == '2'][['From', 'To']]
        colocation_misinfo = misinformation_df.loc[misinformation_df['Direction'] == '1'][['From', 'To']]
        # Finiding the seed and affected nodes
        initialSeed = seed_df.iloc[seed_df.index[seed_df['Sample'] == sample_key][0]]['SeedSet']
        malicious_proj_init = list(set(list(colocation_misinfo['From'])))
        malicious_rec_init = list(set(list(social_misinfo['From'])))
        infect_proj_init = list(set(list(colocation_misinfo['To'])))
        infect_rec_init = list(set(list(social_misinfo['To'])))

        malicious_proj = list(set(malicious_proj_init).union(set(initialSeed)) - set(infect_proj_init))
        infect_proj = list(set(infect_proj_init).union(set(malicious_proj)))
        malicious_rec = list(set(malicious_rec_init).union(set(initialSeed)) - set(infect_rec_init))
        infect_rec = list(set(infect_rec_init).union(set(malicious_rec)))

        seed_vector_proj = [1 if node in malicious_proj else 0 for node in G_proj_org.nodes()]
        inf_vec_all_proj = [1 if node in infect_proj else 0 for node in G_proj_org.nodes()]
        samples_proj.append([seed_vector_proj, inf_vec_all_proj])

        seed_vector_rec = [1 if node in malicious_rec else 0 for node in G_received_org.nodes()]
        inf_vec_all_rec = [1 if node in infect_rec else 0 for node in G_received_org.nodes()]
        samples_rec.append([seed_vector_rec, inf_vec_all_rec])
    ## Changing shape from [samples, 2, nodes] to [samples, nodes, 2]
    samples_proj = torch.Tensor(samples_proj).permute(0, 2, 1)
    samples_received = torch.Tensor(samples_rec).permute(0, 2, 1)

    # %% Saving file
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
    saving_file = '{}_{}.SG'.format(dataset, sample_key)
    with open(saving_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print("Data saved: {}".format(saving_file))

if __name__ == '__main__':

    root_directory = '../data/geosocial_raw/'

    # Simulation A
    net_A = "2500_5"
    dataset_A_0 = "../data/colocation2social_SimA2SimA_0"
    dataset_A_1 = "../data/colocation2social_SimA2SimA_10"
    sim_data_generation(root_directory = root_directory, net_t = net_A, dataset = dataset_A_0, initial_seed_only=True, crop_limit = 1)
    sim_data_generation(root_directory = root_directory, net_t = net_A, dataset = dataset_A_1, initial_seed_only=False, crop_limit = 98)
    #
    # # Simulation B
    # net_B = "4000_5"
    # dataset_B_0 = "../data/colocation2social_SimB2SimB_0"
    # dataset_B_1 = "../data/colocation2social_SimB2SimB_10"
    # sim_data_generation(root_directory = root_directory, net_t = net_B, dataset = dataset_B_0, initial_seed_only=True)
    # sim_data_generation(root_directory = root_directory, net_t = net_B, dataset = dataset_B_1, initial_seed_only=False)