'''
To generate a new token, open this "https://github.com/settings/tokens/new", and you will create the token
'''
import re
import requests
import json
import numpy as np
import networkx as nx
import pickle
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch
import time
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle

def initial_files():
    all_repo = []
    with open('../data_preprocess/dependency_data.txt', encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '\n' or len(line) == 0:
                continue
            splits = line.split(' ')

            if len(splits) == 2:
                repo = splits[0][:-1]
                sid = splits[1]
                # double check
                match = re.findall('https://github\.com/[^"].+', repo)
                if len(match) == 0:
                    continue
                if repo[-1] == ':':
                    #                 print(repo)
                    repo = repo[0:-1]
                all_repo.append(repo)

            elif len(splits) == 3:
                package = splits[1][:-1]
                repo = splits[2]
                # double check
                match = re.findall('https://github\.com/[^"].+', repo)
                if len(match) == 0:
                    continue
                if repo[-1] == ':':
                    #                 print(repo)
                    repo = repo[0:-1]
                all_repo.append(repo)
    all_repo = set(all_repo)
    print(len(all_repo))
    with open('../data_preprocess/all_repo.pkl', 'wb') as fp:
        pickle.dump(list(all_repo), fp)
    with open('../data_preprocess/all_repo_1.pkl', 'wb') as fp:
        pickle.dump(list(all_repo)[0:100000], fp)
    with open('../data_preprocess/all_repo_2.pkl', 'wb') as fp:
        pickle.dump(list(all_repo)[100000:200000], fp)
    with open('../data_preprocess/all_repo_3.pkl', 'wb') as fp:
        pickle.dump(list(all_repo)[200000:300000], fp)
    with open('../data_preprocess/all_repo_4.pkl', 'wb') as fp:
        pickle.dump(list(all_repo)[300000:400000], fp)
    with open('../data_preprocess/all_repo_5.pkl', 'wb') as fp:
        pickle.dump(list(all_repo)[400000:500000], fp)
    with open('../data_preprocess/all_repo_6.pkl', 'wb') as fp:
        pickle.dump(list(all_repo)[500000:], fp)

    # initialize github repo-id graph
    repo_gid_graph = {}
    with open('../data_preprocess/repo_id.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            repo_gid_graph[line[0]] = line[1]
    with open('../data_preprocess/repo_gid_graph.pkl', 'wb') as fp:
        pickle.dump(repo_gid_graph, fp)

def update_files():

    ## Not None
    with open("../data_preprocess/static_features_not_none_1.json") as json_file:
        static_features1 = json.load(json_file)
    with open("../data_preprocess/static_features_not_none_2.json") as json_file:
        static_features2 = json.load(json_file)
    with open("../data_preprocess/static_features_not_none_3.json") as json_file:
        static_features3 = json.load(json_file)
    with open("../data_preprocess/static_features_not_none_4.json") as json_file:
        static_features4 = json.load(json_file)
    with open("../data_preprocess/static_features_not_none_5.json") as json_file:
        static_features5 = json.load(json_file)
    with open("../data_preprocess/static_features_not_none_6.json") as json_file:
        static_features6 = json.load(json_file)
    static_features_not_none = {**static_features1, **static_features2, **static_features3,
                                **static_features4, **static_features5, **static_features6}
    print(len(list(static_features_not_none.keys())))
    with open("../data_preprocess/static_features_not_none.json", "w") as outfile:
        json.dump(static_features_not_none, outfile, indent=4)

    ## All
    with open("../data_preprocess/static_features_1.json") as json_file:
        static_features_none_1 = json.load(json_file)
    with open("../data_preprocess/static_features_2.json") as json_file:
        static_features_none_2 = json.load(json_file)
    with open("../data_preprocess/static_features_3.json") as json_file:
        static_features_none_3 = json.load(json_file)
    with open("../data_preprocess/static_features_4.json") as json_file:
        static_features_none_4 = json.load(json_file)
    with open("../data_preprocess/static_features_5.json") as json_file:
        static_features_none_5 = json.load(json_file)
    with open("../data_preprocess/static_features_6.json") as json_file:
        static_features_none_6 = json.load(json_file)
    static_features_all = {**static_features_none_1, **static_features_none_2, **static_features_none_3,
                           **static_features_none_4, **static_features_none_5, **static_features_none_6}
    with open("../data_preprocess/static_features_all.json", "w") as outfile:
        json.dump(static_features_all, outfile, indent=4)

    ## None
    static_features_none = {k: v for k, v in static_features_all.items() if v is None}
    print(len(list(static_features_none.keys())))
    with open("../data_preprocess/static_features_none.json", "w") as outfile:
        json.dump(static_features_none, outfile, indent=4)

def stackGraph():
    with open("../data_preprocess/static_features_not_none.json", ) as inputFile:
        sf_df = pd.read_json(inputFile, orient='index')
    sf_df.index.names = ['git_id']
    sf_df = sf_df.reset_index()
    # sf_df.head()
    proj2recived_file = '../data_preprocess/gid_sid.txt'
    df_proj2recived = pd.read_csv(proj2recived_file, delim_whitespace=True, header=None)
    df_proj2recived = df_proj2recived.set_axis(['git_id', 'stack_id'], axis=1)
    # df_proj2recived.head()
    # GS_df = pd.merge(sf_df, df_proj2recived, on='git_id') # Keeping only the found values
    GS_df = pd.merge(sf_df, df_proj2recived, on='git_id', how='left')  # Keeping the values not found also
    GS_df['stack_id'] = GS_df['stack_id'].astype('Int64')
    GS_df['created_at'] = pd.to_datetime(GS_df['created_at'])
    # print(len(GS_df.index))
    # print(len(GS_df[GS_df["stack_id"].isnull()].index)) # Checking the number of NaN in stack_id
    # GS_df.to_csv('../CINA_data/gitHub/stack.csv', index = True)
    # GS_df[~GS_df['stack_id'].isna()][['git_id', 'stack_id']].to_csv('../CINA_data/gitHub/gid_sid_pruned.txt', sep='\t',
    #                                                                 index=False)
    # GS_df.head()
    time_df = pd.pivot_table(GS_df, index=GS_df.created_at.dt.month, columns=GS_df.created_at.dt.year,
                             values='stack_id', aggfunc='sum')
    # time_df.to_csv('../CINA_data/gitHub/stack_time.csv', index = True)
    # time_df.plot()
    ax_gs = GS_df.groupby(['year']).count().plot.bar(y=["git_id", "stack_id"])
    for p in ax_gs.patches:
        ax_gs.annotate(f'{p.get_height():0.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                    va='center', xytext=(0, 10), textcoords='offset points')
    # (GS_df['created_at'].dt.month)
    # df = pd.read_csv('sales.csv')

    GS_df['created_at'] = pd.to_datetime(GS_df['created_at'])
    GS_df['year'] = GS_df['created_at'].dt.year
    GS_df['month'] = GS_df['created_at'].dt.month

    # grouped = GS_df.groupby(['year', 'month'])
    grouped = GS_df.groupby(['year'])
    counts = grouped.size()
    df_year = counts.reset_index(name='count')
    # print(df_counts)
    # df_counts.plot()
    ax = df_year.plot.bar(x='year', rot=0, title='Distribution', figsize=(15, 10), fontsize=12)
    # for container in ax.containers:
    #     ax.bar_label(container)
    # colors = ['#5cb85c', '#5bc0de', '#d9534f']
    for p in ax.patches:
        ax.annotate(f'{p.get_height():0.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                    va='center', xytext=(0, 10), textcoords='offset points')
    node_file_stack = '../data_preprocess/sid_all.txt'
    edge_file_stack = '../data_preprocess/all_sid_link_edges.txt'
    G_stack = text2graph1(node_file_stack, edge_file_stack)
    stack_nodes = GS_df.loc[GS_df['stack_id'].notnull(), ['stack_id' ]]['stack_id' ].tolist()
    final_stack_nodes = []
    for node in set(stack_nodes):
        final_stack_nodes.extend(list(nx.single_source_shortest_path_length(G_stack, 14220321, cutoff=2).keys()))
    final_stack_nodes = list(set(final_stack_nodes))  # 298285864 -> 20231
    G_stack_pruned = G_stack.subgraph(final_stack_nodes)
    with open('../data_preprocess/G_stack_pruned.pkl', 'wb') as fp:
        pickle.dump(G_stack_pruned, fp)

def get_github_repo_info_pre(repo_url, user_token=None):
    api_url = f"https://api.github.com/repos/{repo_url[len('https://github.com/'):]}"
    if user_token == None:
        response = requests.get(api_url)
    else:
        GITHUB_API_TOKEN = user_token
        GITHUB_HEADERS = {
            'Authorization': "token " + GITHUB_API_TOKEN,
        }
        response = requests.get(api_url, headers=GITHUB_HEADERS)

    if response.status_code == 200:
        try:
            repo_data = response.json()
            stars = repo_data.get('stargazers_count', 0)
            watchers = repo_data.get('subscribers_count', 0)
            forks = repo_data.get('forks_count', 0)
            if user_token == None:
                contributors_response = requests.get(f"{api_url}/contributors")
            else:
                contributors_response = requests.get(f"{api_url}/contributors", headers=GITHUB_HEADERS)
            if contributors_response.status_code == 200:
                contributors = len(contributors_response.json())
            else:
                contributors = 0

            if user_token == None:
                languages_response = requests.get(f"{api_url}/languages")
            else:
                languages_response = requests.get(f"{api_url}/languages", headers=GITHUB_HEADERS)
            if languages_response.status_code == 200:
                languages = list(languages_response.json().keys())
            else:
                languages = []

            return {
                'url': repo_url,
                'stars': stars,
                'watchers': watchers,
                'forks': forks,
                'contributors': contributors,
                'languages': languages
            }
        except Exception as e:
            print(e)
            return None

    else:
        # print(f"Error fetching data for {repo_url}: {response.status_code}")
        return None

def get_github_repo_info(repo_url, user_token=None):
    api_url = f"https://api.github.com/repos/{repo_url[len('https://github.com/'):]}"
    if user_token == None:
        response = requests.get(api_url)
    else:
        GITHUB_API_TOKEN = user_token
        GITHUB_HEADERS = {
            'Authorization': "token " + GITHUB_API_TOKEN,
        }
        response = requests.get(api_url, headers=GITHUB_HEADERS)

    if response.status_code == 200:
        try:
            repo_data = response.json()
            created_at = repo_data.get('created_at', 0).split('T')[0]
            stars = repo_data.get('stargazers_count', 0)
            subscribers = repo_data.get('subscribers_count', 0)
            forks = repo_data.get('forks_count', 0)
            if user_token == None:
                contributors_response = requests.get(f"{api_url}/contributors")
            else:
                contributors_response = requests.get(f"{api_url}/contributors", headers=GITHUB_HEADERS)
            if contributors_response.status_code == 200:
                contributors = len(contributors_response.json())
            else:
                contributors = 0

            if user_token == None:
                languages_response = requests.get(f"{api_url}/languages")
            else:
                languages_response = requests.get(f"{api_url}/languages", headers=GITHUB_HEADERS)
            if languages_response.status_code == 200:
                languages = list(languages_response.json().keys())
            else:
                languages = []

            return {
                'url': repo_url,
                'created_at': created_at,
                'stars': stars,
                'subscribers': subscribers,
                'forks': forks,
                'contributors': contributors,
                'languages': languages
            }
        except Exception as e:
            print(e)
            return None

    else:
        # print(f"Error fetching data for {repo_url}: {response.status_code}")
        return None

def git_static_features_gen():
    # all_repo = set()
    all_repo = []
    with open('../data_preprocess/dependency_data.txt', encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '\n' or len(line) == 0:
                continue
            splits = line.split(' ')

            if len(splits) == 2:
                repo = splits[0][:-1]
                sid = splits[1]
                # double check
                match = re.findall('https://github\.com/[^"].+', repo)
                if len(match) == 0:
                    continue
                if repo[-1] == ':':
                    #                 print(repo)
                    repo = repo[0:-1]
                all_repo.append(repo)

            elif len(splits) == 3:
                package = splits[1][:-1]
                repo = splits[2]
                # double check
                match = re.findall('https://github\.com/[^"].+', repo)
                if len(match) == 0:
                    continue
                if repo[-1] == ':':
                    #print(repo)
                    repo = repo[0:-1]
                all_repo.append(repo)
    all_repo = set(all_repo)
    print(len(all_repo))

    # initialize github repo-id graph
    repo_gid_graph = {}
    with open('../CINA_data/github/repo_id.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            repo_gid_graph[line[0]] = line[1]
    # token = input("Enter your API key: ")
    token = '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'
    static_features = {}
    repo_count = 0
    extract_tic = time.perf_counter()
    for repo in tqdm(all_repo):
        try:
            static_features[repo_gid_graph[repo]] = get_github_repo_info(repo, token)
        except:
            pass
        repo_count += 1
        if repo_count % 10000 == 0: #50000
            extract_split_toc = time.perf_counter()
            print("Total Extracted {} with {} sec".format(repo_count, extract_split_toc - extract_tic))
            filename_temp = "static_features_{}.json".format(repo_count)
            with open("../CINA_data/github/" + filename_temp, "w") as outfile:
                json.dump(static_features, outfile, indent=4)

        # print(repo)
    print("Github extraction finished for {} within {} sec".format(repo_count, time.perf_counter() - extract_tic))

    with open("../CINA_data/github/static_features.json", "w") as outfile:
        json.dump(static_features, outfile, indent=4)
    # print(static_features)
    # Opening JSON file
    with open('../CINA_data/github/static_features.json') as json_file:
        static_features = json.load(json_file)

    all_val = list(static_features.values())
    print("Total nodes with none features {}".format(all_val.count(None)))  # counting none
    print("Total nodes with features {}".format(len(static_features) - all_val.count(None)))

    static_features_not_none = {k: v for k, v in static_features.items() if v is not None}
    static_features.clear()
    static_features.update(static_features_not_none)
    # static_features_not_none
    # static_features_none = {k: v for k, v in static_features.items() if v is None}
    # static_features_none
    with open("../CINA_data/github/static_features_not_none.json", "w") as outfile:
        json.dump(static_features, outfile, indent=4)

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
    # print(adj_matrix.shape)
    static_array = np.asarray(static_features, dtype=np.float32)
#     nx.draw(G,with_labels=True)
#     plt.draw()
#     plt.show()
    return G, adj_matrix, static_array

def text2graph1(node_file, edge_file):
    nodes_list = pd.read_csv(node_file, delim_whitespace=True, header=None)[0].values.tolist()
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    df = pd.read_csv(edge_file, delim_whitespace=True, header=None)
    for index, row in df.iterrows():
        G.add_edge(row[0], row[1])
#     adj_matrix = nx.to_numpy_array(G, dtype='f')
    return G

def text2graph(node_file, edge_file):
    if node_file !=None:
        nodes_list = pd.read_csv(node_file, delim_whitespace=True, header=None)[0].values.tolist()
        G = nx.Graph()
        G.add_nodes_from(nodes_list)
    else:
        G = nx.Graph()
    df = pd.read_csv(edge_file, delim_whitespace=True, header=None)
    for index, row in df.iterrows():
        G.add_edge(row[0], row[1])
    adj_matrix = nx.to_numpy_array(G, dtype='f')
    return G, adj_matrix

def generate_seed_vector(top_nodes, seed_num, G):
    seed_nodes = random.sample(top_nodes, seed_num)
    seed_vector = [1 if node in seed_nodes else 0 for node in G.nodes()]
    return seed_vector

def infected_nodes(G, seed_vector_init, inf_vec_all, diffusion='LT', diff_num = 10, iter_num = 100):
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

        for j in range(1, len(iterations)):
            node_status.update(iterations[j]['status'])

        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1

        inf_vec_all += inf_vec
    return inf_vec_all

def data_generation(org_graph, adj_matrix, static_array, nums, percentage=10, diffusion='LT', dataset='github'):
    G = nx.from_numpy_matrix(adj_matrix)
    node_num = len(G.nodes())
    seed_num = int(percentage * node_num / 100)
    samples = []

    degree_list = list(G.degree())
    degree_list.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [x[0] for x in degree_list[:int(len(degree_list) * 0.3)]]

    for i in range(nums):
        print('Sample {} generating'.format(i))
        seed_vector = generate_seed_vector(top_nodes, seed_num, G)
        inf_vec_all = torch.zeros(node_num)
        for j in range(10):
            if diffusion == 'LT':
                model = ep.ThresholdModel(G)
                config = mc.Configuration()
                for n in G.nodes():
                    config.add_node_configuration("threshold", n, 0.5)
            elif diffusion == 'IC':
                model = ep.IndependentCascadesModel(G)
                config = mc.Configuration()
                for e in G.edges():
                    config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])
            elif diffusion == 'SIS':
                model = ep.SISModel(G)
                config = mc.Configuration()
                config.add_model_parameter('beta', 0.001)
                config.add_model_parameter('lambda', 0.001)
            else:
                raise ValueError('Only IC, LT and SIS are supported.')

            config.add_model_initial_configuration("Infected", seed_vector)

            model.set_initial_status(config)

            iterations = model.iteration_bunch(100)

            node_status = iterations[0]['status']

            for j in range(1, len(iterations)):
                node_status.update(iterations[j]['status'])

            inf_vec = np.array(list(node_status.values()))
            inf_vec[inf_vec == 2] = 1

            inf_vec_all += inf_vec

        inf_vec_all = inf_vec_all / 10
        samples.append([seed_vector, inf_vec_all])

    samples = torch.Tensor(samples).permute(0, 2, 1) ## Why permutation
    f = open('{}_mean_{}{}.SG'.format(dataset, diffusion, percentage), 'wb')
    pickle.dump({'adj': adj_matrix, 'inverse_pairs': samples, 'prob':adj_matrix, 'original_graph':org_graph, 'static_features': static_array}, f)
    f.close()
    print('Data generation finished')


def cross_data_generation(G_proj_org, adj_proj, static_proj, G_received_org, adj_received, proj2recived_file,
                          nums=100, percentage=10, diffusion_proj='LT', diffusion_recived='IC', dataset='github2stack'):
    nodes_name_G_proj = np.array(list(G_proj_org.nodes()))
    nodes_name_G_recived = np.array(list(G_received_org.nodes()))
    df_proj2recived = pd.read_csv(proj2recived_file, delim_whitespace=True, header=None)
    proj_nodes = df_proj2recived[0].to_numpy() #0
    receipient_nodes = df_proj2recived[1].to_numpy()#1

    G_proj = nx.from_numpy_matrix(adj_proj)
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

    G_received = nx.from_numpy_matrix(adj_received)
    node_num_received = len(G_received.nodes())
    # seed_num_received = int(percentage * node_num_received / 100)
    samples_received = []

    for j in range(nums):
        print('Sample {} generating'.format(j))
        seed_vector_proj = generate_seed_vector(top_nodes_proj, seed_num_proj, G_proj)
        inf_vec_all_proj = torch.zeros(node_num_proj)
        inf_vec_all_proj = infected_nodes(G_proj, seed_vector_proj, inf_vec_all_proj, diffusion=diffusion_proj,
                                          diff_num=10, iter_num=100)
        inf_vec_all_proj = inf_vec_all_proj / 10  # divided by the diffusion_num
        samples_proj.append([seed_vector_proj, inf_vec_all_proj])

        inf_proj_idx = []
        for i in nodes_name_G_proj[inf_vec_all_proj == 1]:
            inf_proj_idx.extend(np.where(proj_nodes == i)[0].tolist())
        # seed_name_received = nodes_name_G_recived[inf_proj_idx]
        seed_name_received = receipient_nodes[inf_proj_idx]
        seed_vector_received = []
        for index, element in enumerate(nodes_name_G_recived):
            seed_vector_received.append(1) if element in seed_name_received else seed_vector_received.append(0)
        inf_vec_all_received = torch.zeros(node_num_received)
        inf_vec_all_received = infected_nodes(G_received, seed_vector_received, inf_vec_all_received,
                                              diffusion=diffusion_recived, diff_num=10, iter_num=100)
        inf_vec_all_received = inf_vec_all_received / 10  # divided by the diffusion_num
        samples_received.append([seed_vector_received, inf_vec_all_received])

    samples_proj = torch.Tensor(samples_proj).permute(0, 2, 1)  ## Why permutation: Changing shape from [samples, 2, nodes] to [samples, nodes, 2]
    samples_received = torch.Tensor(samples_received).permute(0, 2, 1)  ## Why permutation

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
def main():
    ############## Step 1 ###################
    # initial_files()
    ############## Step 2 ###################
    with open('../data_preprocess/repo_gid_graph.pkl', 'rb') as fp:
        repo_gid_graph = pickle.load(fp)  # Use it only for the first time run
    # with open('../data_preprocess/repo_gid_graph_updated.pkl', 'rb') as fp:
    #     repo_gid_graph = pickle.load(fp)

    with open('../data_preprocess/all_repo_1.pkl', 'rb') as fp:  # run it for 1-6 files
        all_repo_test = pickle.load(fp)
    static_features_file = '../data_preprocess/static_features_1.json'
    static_features_not_none_file = "../data_preprocess/static_features_not_1.json"
    repo_id_file = '../data_preprocess/repo_id.txt'
    git_static_features_gen(all_repo_test, repo_gid_graph, static_features_file, static_features_not_none_file)

    ############## Step 3 ###################
    update_files()
    stackGraph()
    ############## Step 3 ###################
    features_file_git = '../data_preprocess/static_features_not_none.json'
    edges_file_git = '../data_preprocess/gid_edges.txt'
    # G_git, adj_matrix_git, static_array_git = json2graph(features_file_git, edges_file_git)
    # data_generation(G_git, adj_matrix_git, static_array_git, 100, percentage=10, diffusion='LT', dataset='../CINA_data/github/github')

    node_file_stack = '../data_preprocess/sid_all.txt'
    # edge_file_stack = '../data_preprocess/all_sid_link_edges.txt'
    # G_stack, adj_matrix_stack = text2graph(node_file_stack, edge_file_stack)
    with open('../CINA_data/gitHub/G_stack_pruned.pkl', 'rb') as fp:
        G_stack = pickle.load(fp)
    adj_matrix_stack = nx.to_numpy_array(G_stack, dtype='f')
    features_file_git = '../data_preprocess/static_features_not_none.json'
    edges_file_git = '../data_preprocess/gid_edges.txt'
    G_git, adj_matrix_git, static_array_git = json2graph(features_file_git, edges_file_git)

    proj2recived_file = '../data_preprocess/gid_sid_pruned.txt'
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=1, percentage=10, diffusion_proj='LT', diffusion_recived='IC',
                          dataset='../data/github2stack')


if __name__ == "__main__":
    ############# Testing Purpose ############
    features_file_git = '../data_preprocess/github_pre_analysis_data/static_features_not_none.json'
    edges_file_git = '../data_preprocess/github_pre_analysis_data/gid_edges.txt'
    G_git, adj_matrix_git, static_array_git = json2graph(features_file_git, edges_file_git)

    node_file_stack = '../data_preprocess/stack_pre_analysis_data/sid_all.txt'
    edge_file_stack = '../data_preprocess/stack_pre_analysis_data/sid_link_edges.txt'
    G_stack, adj_matrix_stack = text2graph(node_file_stack, edge_file_stack)
    proj2recived_file = '../data_preprocess/github_pre_analysis_data/gid_sid.txt'
    cross_data_generation(G_proj_org=G_git, adj_proj=adj_matrix_git, static_proj=static_array_git,
                          G_received_org=G_stack, adj_received=adj_matrix_stack, proj2recived_file=proj2recived_file,
                          nums=1, percentage=10, diffusion_proj='LT', diffusion_recived='LT',
                          dataset='../data/github2stackTest1')