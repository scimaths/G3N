import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import random
import subgraph


class GNNLayer(nn.Module):
    
    def __init__(self, in_features, out_features, params, hidden_features, sampling_prob = 1, scale_mlp = False) -> None:
        super(GNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.keys = params['keys']

        self.d = params['d']
        self.t = params['t']
        self.scalar = params['scalar']
        self.combination = params['combination']

        self.one_hot_length = subgraph.one_hot_length(self.t)
        self.scale = scale_mlp
        if self.scale:
            self.scaling_mlp = nn.ModuleDict()
            self.scaling_embed = nn.ModuleDict()
        if self.scalar:
            self.eps = nn.ParameterDict()
        self.attn_transform = nn.ModuleDict()
        self.transform = nn.ModuleDict()
        self.scaling = nn.ModuleDict()
        for key in self.keys: #keys are the f_pos values
            k = str(key)
            if self.combination == "multi":
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.Sigmoid())
            elif self.combination == "sum":
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.ReLU())
            self.attn_transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.ReLU())
            if self.scaling:
                self.scaling_mlp[k] = torch.nn.Sequentia;(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.Sigmoid())
                self.scaling_embed[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.Tanh())
            if self.scalar:
                self.eps[k] = torch.nn.Parameter(torch.Tensor([0]))
        self.attn_linear = nn.Linear(self.in_features, self.hidden_features)
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.sampling_prob = sampling_prob
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    
    def forward(self, h, pair_info, sampling_prob=1):

        # transform roots
        h3 = self.linear(h)
        h_attn_emb = self.attn_linear(h) #N x hid
        pairs, degrees, scatter = pair_info

        for key in pairs:
            if len(scatter[key]) == 0:
                continue

            k = str(key)
            centre_node_ind = torch.tensor(scatter[key]).long() #n_key
            nbr_node_ind = torch.tensor(pairs[key]).long() # t x n_key
            nbr_node_emb = h[nbr_node_ind] #t x n_key x d
            nbr_node_emb_attn = self.attn_transform[k](nbr_node_emb) #t x n_key x hid
            nbr_node_emb_attn = torch.transpose(nbr_node_emb_attn, 0, 1) #n_key x t x hid
            centre_node_emb_attn = torch.unsqueeze(h_attn_emb[centre_node_ind], -1) #n_key x hid x 1
            sim_scores_dist = torch.squeeze(torch.matmul(nbr_node_emb_attn, centre_node_emb_attn)) #n_key x t
            attn_scores_dist = F.softmax(sim_scores_dist, dim = -1) #n_key x t
            attn_scores_dist = torch.transpose(attn_scores_dist, 0, 1) #t x n_key
            
            print('given positional encoding fpos ', key)
            if self.combination == "multi":  # s(h_x @ W) * s(h_y @ W)
                h_temp = 1
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i]))
                    print('h_t shape in GNN layer ', h_t.shape)
                    print('sample one hot degree ', degrees[key][i])
                    if not self.scaling:
                        h_temp = h_temp * self.transform[k](h_t) *attn_scores_dist[i]
                    else:
                        h_temp = h_temp * self.scaling_mlp[k](h_t) * self.scaling_embed[k](h_t)
                print('h temp shape in GNN layer ', h_temp.shape)
            elif self.combination == "sum":  # s(h_x @ W + h_y @ W)
                h_temp = 0
                for i in range(self.t):
                    h_t = torch.hstack((h[pairs[key][i]], degrees[key][i]))
                    if not self.scaling:
                        h_temp = h_temp + h_t*attn_scores_dist[i]
                    else:
                        h_temp += self.scaling_mlp[k](h_t) * self.scaling_embed[k](h_t)
                h_temp = self.transform[k](h_temp)

            h_sum = torch.zeros((h.shape[0], self.out_features)).to(self.dummy_param.device)
            print('central nodes scatter key ', scatter[key])
            scatter_add(src=h_temp, out=h_sum, index=scatter[key], dim=0)

            if self.scalar:
                h_sum = (1 + self.eps[k]) * h_sum
            unif = random.random()
            if unif<sampling_prob:
                h3 = h3 + h_sum

        return h3