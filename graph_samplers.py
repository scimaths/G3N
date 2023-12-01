import networkx as nx
import numpy as np
import scipy.sparse as sp

class EdgeSampler:
    def __init__(self, edge_list, num_nodes, deg_seq):
        self.edge_list = edge_list
        self.edges = np.array(edge_list, dtype = np.int32)
        self.num_nodes = num_nodes
        self.deg_seq = deg_seq
        self.edge_num = len(self.edges)
        self.edge_probs = self.edge_probability()

    def edge_probability(self):
        un_norm_probs = 1/self.deg_seq[self.edges[:, 0]] + 1/self.deg_seq[self.edges[:, 1]]
        probs = un_norm_probs/self.num_nodes
        return probs
    
    def sample_nodes(self, budget):
        edge_iter = np.arange(self.edge_num, dtype = np.int32)
        sampled_edges = np.random.choice(edge_iter, size = budget, p = self.edge_probs)
        unique_sampled_edges = np.unique(sampled_edges)
        edges = self.edges[unique_sampled_edges]
        unique_nodes = np.unique(edges.flatten())
        return unique_nodes
    
class NodeSampler:
    def __init__(self, adj_mat, num_nodes, deg_seq):
        self.adj = adj_mat
        self.num_nodes = num_nodes
        self.degrees = deg_seq
        self.norm_adj = self.normalized_adjacency()
        self.node_samp_prob = self.node_probability()
        
    def normalized_adjacency(self):
        inv_deg_seq = np.divide(np.ones(self.num_nodes), self.degrees, where = self.degrees!=0)
        inv_deg_diag = sp.diags(inv_deg_seq, shape = (self.num_nodes, self.num_nodes))
        return inv_deg_diag@self.adj
    
    def node_probability(self):
        A_sq = self.norm_adj*self.norm_adj
        node_probs = A_sq.sum(0)
        node_probs = node_probs/(node_probs.sum())
        node_probs = sp.lil_matrix(node_probs).toarray().flatten()
        return node_probs
    
    def sample_nodes(self, budget):
        nodes = np.arange(self.num_nodes, dtype = np.int32)
        sampled_nodes = np.random.choice(nodes, size = budget, p = self.node_samp_prob)
        unique_nodes = np.unique(sampled_nodes)
        return unique_nodes
    
class RWSampler:
    def __init__(self, adj_list, num_nodes):
        self.adj_list = adj_list
        self.num_nodes = num_nodes

    def sample_nodes(self, init_size, walk_len):
        nodes = np.arange(self.num_nodes)
        rw_starts = np.random.choice(nodes, size=init_size)
        rw_explored = set(rw_starts)
        for v in rw_starts:
            start = v
            explored = [v]
            for step in range(walk_len):
                start = np.random.choice(self.adj_list[start], size = 1)[0]
                explored.append(start)
            rw_explored = rw_explored.union(set(explored))
        return list(rw_explored)

class MRWSampler:
    def __init__(self, adj_list, num_nodes, deg_seq):
        self.adj_list = adj_list
        self.num_nodes = num_nodes
        self.deg_seq = np.array(deg_seq)

    def sample_next_node(self, node_set):
        node_list = np.array(list(node_set), dtype = np.int32)
        node_probs = self.deg_seq[node_list]
        node_probs = node_probs/np.sum(node_probs)
        sampled_node = np.random.choice(node_list, size=1, p = node_probs)[0]
        return sampled_node

    def sample_nodes(self, init_size, budget):
        nodes = np.arange(self.num_nodes)
        rw_starts = np.random.choice(nodes, size=init_size)
        rw_starts = set(rw_starts)
        visited_nodes = rw_starts.copy()
        for i in range(init_size, budget):
            start = self.sample_next_node(rw_starts)
            next_node = np.random.choice(self.adj_list[start], size=1)[0]
            visited_nodes.update([next_node])
            rw_starts.remove(start)
            rw_starts.update([next_node])
        return visited_nodes


if __name__=='__main__':
    G = nx.erdos_renyi_graph(50, 0.5)
    A = nx.adjacency_matrix(G)
    adj_list = list(map(list, iter(G.adj.values())))
    print('adjacency list ', adj_list)
    degrees = np.array(G.degree)[:, 1]
    mrwsampler = MRWSampler(adj_list, 50, degrees)
    sampled_nodes = mrwsampler.sample_nodes(5, 15)
    G_in = nx.induced_subgraph(G, sampled_nodes)
    print('sampled edges ', G_in.edges)
    print('sampled nodes ', len(G_in.nodes))
    print('degrees ', G_in.degree)

