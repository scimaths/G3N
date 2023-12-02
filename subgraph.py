import torch
import networkx as nx
import itertools
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from pynauty import *
import numpy as np
import random
import graph_samplers as gs

def one_hot_length(t, get_deg=False):
    max_deg = 5 
    iso_type = 2 if t==3 else 0
    if get_deg:
        return max_deg + iso_type, max_deg # 7(if t=3) else 5, 5
    else:
        return max_deg + iso_type # 7(if t=3) else 5

def get_avg_diameter(graph_collections):
    diameters = []
    for graphs in graph_collections:
        for graph in graphs:
            G = to_networkx(graph).to_undirected()
            for g in nx.connected_components(G):  # usually batched graphs
                diameters.append(nx.diameter(G.subgraph(g)))
    avg_diam = np.mean(diameters)
    print('average graph diameter:', avg_diam)
    return avg_diam

def get_keys_from_loaders(loaders):
    keys = set()
    for loader in loaders:
        for batch in loader:
            print('batch in loader ', batch)
            pairs, _, _ = batch.pair_info[0]
            for key in pairs:
                keys.add(key)
    print('topology types:', keys) #keys is most probably f_pos, i.e increasing seq of dist from central node
    print('number of topology types:', len(keys))
    return keys

def transform(data, d, t, connected, num_samp = 10000, samp_frac = 1, sampler = None, 
              budget = 50, init_size = 20, walk_len = 20):
    data.pair_info, sampled_nodes = compute_nhbr_pair_data(to_networkx(data), d, t, connected, num_samp, samp_frac, 
                                            sampler, budget, init_size, walk_len)
    
    if sampler is None:
        return data
    else:
        return data, sampled_nodes

def knbrs(G_in, start, k, num_samples=10000):  # get k-hop neighbourhood
    # nbrs = nx.single_source_shortest_path_length(G_in,source=start,cutoff=k).keys()  # slightly slower than below?
    nbrs = set([start])
    for _ in range(1, k+1):
        #next_nbrs = set((nbr for n in nbrs for nbr in G_in[n]))
        next_nbrs = set([])
        for n in nbrs:
            sampled_nbrhood = set(random.sample(G_in[n])) if len(G_in[n])>=num_samples else set(G_in[n])
            next_nbrs = next_nbrs.union(sampled_nbrhood)
        nbrs = nbrs.union(next_nbrs)
    return nbrs

def induced_degree(u, G, subgraph_nodes):
    ret = len(subgraph_nodes.intersection(set(G[u])))
    return ret

def original_degree(u, G):
    ret = len(set(G[u]))
    return ret

def compute_nhbr_pair_data(G, d, t, require_connected, num_sample = 100, samp_frac = 1, sampler = None, 
                           budget = 50, init_size = 20, walk_len = 20):
    G = G.to_undirected()  # works assuming undirected
    sampled_nodes = len(G.nodes)
    if sampler == 'edge':
        g_edge_list = np.array(G.edges)
        g_num_nodes = len(G.nodes)
        g_degrees = np.array(G.degree, dtype = np.int32)[:, 1]
        edge_sampler = gs.EdgeSampler(g_edge_list, g_num_nodes, g_degrees)
        sampled_nodes = edge_sampler.sample_nodes(budget)
        G = nx.induced_subgraph(G, sampled_nodes)
    elif sampler == 'node':
        g_adj_mat = nx.adjacency_matrix(G)
        g_num_nodes = len(G.nodes)
        g_degrees = np.arrray(G.degree, dtype = np.int32)[:, 1]
        node_sampler = gs.NodeSampler(g_adj_mat, g_num_nodes, g_degrees)
        sampled_nodes = node_sampler.sample_nodes(budget)
        G = nx.induced_subgraph(G, sampled_nodes)
    elif sampler == 'rw':
        g_adj_list = list(map(list, iter(G.adj.values())))
        g_num_nodes = len(G.nodes)
        rw_sampler = gs.RWSampler(g_adj_list, g_num_nodes)
        sampled_nodes = rw_sampler.sample_nodes(init_size, walk_len)
        G = nx.induced_subgraph(G, sampled_nodes)
    elif sampler == 'mrw':
        g_adj_list = list(map(list, iter(G.adj.values())))
        g_num_nodes = len(G.nodes)
        g_degrees = np.array(G.degree)[:, 1]
        mrw_sampler = gs.MRWSampler(g_adj_list, g_num_nodes, g_degrees)
        sampled_nodes = mrw_sampler.sample_nodes(init_size, budget)
        G = nx.induced_subgraph(G, sampled_nodes)
    
    pairs = {} #dict whose keys are f_pos, pairs[pos] is a list of lists. ith list contains all nodes coming at ith in dist from centre node in some sampled graph with that pos value
    degrees = {} #some one hot degree based on f_pos and ith node from central, each entry is list of list
    scatter = {}#dict whole keys are f_pos. scatter[pos] is list of central nodes from which scattering/diffusion occurs with that particular pos value
    iso_hash = {} #dict containing all iso types amongst t-ord subgraphs across all d-hops of all nodes, basically f_iso for hash value
    distances = dict(nx.all_pairs_shortest_path_length(G))

    one_hot_len, max_deg = one_hot_length(t, get_deg=True) # t-> order

    # create pair_neighbourhood
    #print(" Nodes in one batch(?) shape ", len(G.nodes))
    #print("Nodes in one batch(?) ", G.nodes)
    for node in G.nodes:  # sorted 0,1,.. etc
        #for a node, search it's d-hop neighbourhood
        subgraph_nodes = knbrs(G, node, d, num_sample) #list of all nodes reachable by atmost k hop from u
        if t==1:
            subgraph_nodes.remove(node)
        for comb in itertools.combinations(subgraph_nodes, t): # all possible t combinations of graph nodes
            unif = random.random()
            if unif<samp_frac:
                if t == 1:
                    is_connected = True
                    edges = 0
                elif t==2:
                    u, v = comb
                    is_connected = (u, v) in G.edges
                    edges = int(is_connected)
                elif t==3:
                    u, v, w = comb
                    edges = int((u, v) in G.edges) + int((u, w) in G.edges) + int((w, v) in G.edges)
                    is_connected = edges >= 2
                    iso_type = edges % 2
                else:
                    # obtain induced subgraph
                    sg = nx.induced_subgraph(G, comb)
                    # construct a pynauty subgraph
                    pynauty_subgraph = Graph(len(comb))
                    # edge list for an induced subgraph
                    edge_list = sg.edges(list(comb)) 
                    # reassign the node ids in induced subgraph to add edges in pynauty subgraph
                    dict_keys = {}
                    for key, value in enumerate(comb):
                        dict_keys[value] = key #making the new graph with nodes 0, 1, ... , t-1
                    # add edges into a pynauty graph
                    for edge_data in edge_list:
                        pynauty_subgraph.connect_vertex(dict_keys[edge_data[0]], [dict_keys[edge_data[1]]])
                    # subgraph signature by canonical label certificate
                    iso_type = hash(int.from_bytes(certificate(pynauty_subgraph), byteorder='big')) #isomorphism hash value of subgraph
                    
                    if(iso_type not in iso_hash):
                        iso_hash[iso_type] = len(iso_hash) #instead of having a 'weird' f_iso, we make it 'nice' from 0,...,I-1 where I is the no of isomorphism classes
                    
                    is_connected = nx.is_connected(sg)
                    iso_type = iso_hash[iso_type] #change the isomorphism class as per convention

                if require_connected and not is_connected:
                    continue

                tuple_list = sorted([(distances[node][u], u) for u in comb], key=lambda x: x[0]) #verices in increasing distance from node

                key = [x[0] for x in tuple_list] #list of distances from u to nodes in the t-ord subgraph comb
                key = tuple(key) #key is basically f_pos for the subgraph comb

                if key not in pairs:
                    pairs[key] = [[] for _ in range(t)]
                    degrees[key] = [[] for _ in range(t)]
                    scatter[key] = []

                for i in range(t): #iterate across nodes in comb
                    u = tuple_list[i][1] #node index
                    deg = induced_degree(u, G=G, subgraph_nodes=subgraph_nodes) #deg of u in induced subgraph of k-hop nbrs of node
                    deg = min(max_deg, deg) - 1
                    one_hot_deg = [0 for _ in range(one_hot_len)] #one hot len 5 or 7
                    one_hot_deg[deg] = 1
                    if t==3:
                        one_hot_deg[max_deg + iso_type] = 1
                    pairs[key][i].append(u) #pairs[key][i] is list of all nodes having f_pos=key and at a (sorted) position i from centraql node
                    degrees[key][i].append(one_hot_deg)
                scatter[key].append(node) #list of central nodes having nbrhd graph of positional type fpos
            else:
                continue
    for key in pairs:
        pairs[key] = torch.tensor(pairs[key]).to('cuda')
        degrees[key] = torch.tensor(degrees[key]).to('cuda')
        scatter[key] = torch.tensor(scatter[key]).long().to('cuda')
    nhbr_info = (pairs, degrees, scatter)
    #print('returning nbrhd info ', nhbr_info)
    return nhbr_info, sampled_nodes