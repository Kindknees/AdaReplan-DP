import torch
import torch.nn as nn
from torch.distributions import Normal
from model.common.mlp import MLP 

class D3PAdaptor(nn.Module):
    def __init__(self, obs_dim, action_dim, output_mean, seq_len=1, chunk_size=4, mlp_dims=[256, 512, 1024, 512, 256]):
        super().__init__()
        
        input_dim = (obs_dim * seq_len) + (action_dim * chunk_size)
        
        dim_list = [input_dim] + mlp_dims
        
        self.net = MLP(
            dim_list=dim_list,
            activation_type="Mish",    
            out_activation_type="Mish",
            use_layernorm=False 
        )

        self.mean_layer = nn.Linear(mlp_dims[-1], 1)

        self.log_std = nn.Parameter(torch.zeros(1))

        nn.init.constant_(self.mean_layer.bias, output_mean)
        nn.init.constant_(self.mean_layer.weight, 0.0)

    def forward(self, obs_dict, noisy_action):
        B = noisy_action.shape[0]
        
        obs_flat = obs_dict["state"].view(B, -1)
        action_flat = noisy_action.view(B, -1)
        
        x = torch.cat([obs_flat, action_flat], dim=-1)
        feat = self.net(x)

        mean = self.mean_layer(feat).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        
        return Normal(mean, std)