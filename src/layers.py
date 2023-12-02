import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import subgraph


class GNNLayer(nn.Module):
    
    def __init__(self, in_features, out_features, params, update_type='gnn', single_mlp=False) -> None:
        super(GNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.keys = list(params['keys'])
        self.key_to_idx = {str(key): idx for idx, key in enumerate(self.keys)}
        self.one_hots = [torch.tensor([int(i==j) for j in range(len(self.keys))]).to('cuda') for i in range(len(self.keys))]

        self.d = params['d']
        self.t = params['t']
        self.scalar = params['scalar']
        self.combination = params['combination']
        self.is_gmn = (update_type == 'gmn')
        self.single_mlp = single_mlp

        self.one_hot_length = subgraph.one_hot_length(self.t)
        if self.single_mlp:
            self.one_hot_length += len(self.keys)

        if self.scalar:
            self.eps = nn.ParameterDict()

        self.transform = nn.ModuleDict()
        if self.is_gmn:
            self.gmn = nn.ModuleDict()
        for key in self.keys:
            k = self.mlp_index(str(key))
            if self.combination == "multi":
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.Sigmoid())
            elif self.combination == "sum":
                self.transform[k] = torch.nn.Sequential(nn.Linear(self.in_features + self.one_hot_length, self.out_features), torch.nn.ReLU())
            if self.is_gmn:
                self.gmn[k] = torch.nn.Sequential(
                    nn.Linear(self.in_features + self.one_hot_length, self.out_features),
                    nn.ReLU(),
                    nn.Linear(self.out_features, self.out_features),
                    nn.Sigmoid()
                )
    
            if self.scalar:
                self.eps[k] = torch.nn.Parameter(torch.Tensor([0]))
            
            if self.single_mlp:
                break

        self.linear = nn.Linear(self.in_features, self.out_features)

        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def mlp_index(self, k):
        if self.single_mlp:
            return "0"
        else:
            return k

    def forward(self, h, pair_info):

        # transform roots
        h3 = self.linear(h)

        pairs, degrees, scatter = pair_info

        for key in pairs:
            if len(scatter[key]) == 0:
                continue

            k = str(key)
            mlp_index = self.mlp_index(k)
            key_idx = self.key_to_idx[k]
            
            if self.combination == "multi":  # s(h_x @ W) * s(h_y @ W)
                h_temp = 1
                for i in range(self.t):
                    degrees_key_i = degrees[key][i]
                    if self.single_mlp:
                        h_t = torch.hstack((h[pairs[key][i]], degrees_key_i, self.one_hots[key_idx].repeat(degrees_key_i.shape[0], 1)))
                    else:
                        h_t = torch.hstack((h[pairs[key][i]], degrees_key_i))
                    if self.is_gmn:
                        transform_res = self.transform[mlp_index](h_t) * self.gmn[mlp_index](h_t)
                    else:
                        transform_res = self.transform[mlp_index](h_t)
                    h_temp = h_temp * transform_res
            elif self.combination == "sum":  # s(h_x @ W + h_y @ W)
                h_temp = 0
                for i in range(self.t):
                    degrees_key_i = degrees[key][i]
                    if self.single_mlp:
                        h_t = torch.hstack((h[pairs[key][i]], degrees_key_i, self.one_hots[key_idx].repeat(degrees_key_i.shape[0], 1)))
                    else:
                        h_t = torch.hstack((h[pairs[key][i]], degrees_key_i))
                    h_temp = h_temp + h_t
                if self.is_gmn:
                    h_temp = self.gmn[mlp_index](h_temp) * self.transform[mlp_index](h_temp)
                else:
                    h_temp = self.transform[mlp_index](h_temp)
                h_temp = transform_res

            h_sum = torch.zeros((h.shape[0], self.out_features)).to(self.dummy_param.device)
            scatter_add(src=h_temp, out=h_sum, index=scatter[key], dim=0)

            if self.scalar:
                h_sum = (1 + self.eps[mlp_index]) * h_sum

            h3 = h3 + h_sum

        return h3