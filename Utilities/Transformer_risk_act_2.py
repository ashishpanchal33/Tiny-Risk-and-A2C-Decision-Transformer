import gymnasium as gym
import os
#import gym
import numpy as np

import collections
import pickle
import tqdm
from stable_baselines3 import PPO


import sys
import random
import csv
from datetime import datetime

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Beta,Categorical

"""
this extremely minimal GPT model is based on:
Misha Laskin's tweet: 
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

the above colab has a bug while applying masked_fill which is fixed in the
following code

"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = layer_init(nn.Linear(h_dim, h_dim), std=0.01)
        self.k_net = layer_init(nn.Linear(h_dim, h_dim), std=0.01)
        self.v_net = layer_init(nn.Linear(h_dim, h_dim), std=0.01)

        self.proj_net = layer_init(nn.Linear(h_dim, h_dim), std=0.01)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                layer_init(nn.Linear(h_dim, 4*h_dim), std=0.01),
                nn.GELU(),
                layer_init(nn.Linear(4*h_dim, h_dim), std=0.01),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x




class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, 
                 n_heads, drop_p, max_timestep=4096,a2_concentration=5):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.a2_concentration = a2_concentration

        ### transformer blocks
        input_seq_len = 4 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = layer_init(torch.nn.Linear(1, h_dim), std=0.01)
        self.embed_state = layer_init(torch.nn.Linear(state_dim, h_dim), std=0.01)
        
        # # discrete actions - maybe i'll not use it for risk... will look into this later
        self.embed_action_1 = torch.nn.Embedding(max_timestep, h_dim) # not act_dim
        self.embed_action_2 = layer_init(torch.nn.Linear(1, h_dim), std=0.01)
        self.embed_action = layer_init( torch.nn.Linear(2*h_dim,h_dim), std=0.01)
        
        self.use_action_tanh = True # False for discrete actions

        # continuous actions
        #self.embed_action = torch.nn.Linear(act_dim, h_dim)
        #use_action_tanh = True # True for continuous actions
        
        ### prediction heads
        #self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_rtg = layer_init(torch.nn.Linear(h_dim, 1#act_dim
                                                             ), std=0.01)
        
        #removing this
        #self.predict_state = layer_init(torch.nn.Linear(h_dim, state_dim), std=0.01)
        
        
        self.predict_action = nn.Sequential(
            *([layer_init(nn.Linear(h_dim, act_dim), std=0.01)] + ([nn.Tanh()] if self.use_action_tanh else []))
        )


        self.predict_actor_1 = layer_init(nn.Linear(h_dim, act_dim), std=0.01)
        self.predict_actor_2 = nn.Sequential( layer_init(nn.Linear(h_dim, 1), std=0.01),nn.GELU(),nn.Sigmoid())


    

        
        
        #self.predict_actor_1 = nn.Sequential( *([layer_init(nn.Linear(h_dim, act_dim), std=0.01)] +   ([nn.Tanh()] if use_action_tanh else []))
                                                        #)

        
        #self.predict_actor_2 = nn.Sequential(layer_init(nn.Linear(h_dim, 2), std=0.01), nn.Softmax(dim=2),)




    


    def forward(self, timesteps, states, actions_1,actions_2, returns_to_go,print_=False,return_logit = False,return_log_prob_a2 = False,return_og_log_prob_a2 = False):

        B, T, _ = states.shape
        #print(timesteps.device
        #     ,states.device
        #     ,actions.device
        #     ,returns_to_go.device)
        time_embeddings = self.embed_timestep(timesteps.long())

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        #print(actions.shape)
        #print(self.embed_action(actions).squeeze().shape)
        #print(time_embeddings.shape)

        action_embeddings_1 = self.embed_action_1(actions_1[:,:,].long() )+ time_embeddings
        action_embeddings_2 = self.embed_action_2(actions_2[:,:,None])+ time_embeddings
        
        
        #embed_action_1
        #embed_action_2
        #embed_action

        if False:#print_:
            print(self.embed_state(states).shape)
            print(self.embed_rtg(returns_to_go.float()).shape,
                  time_embeddings.shape)
            
        #action_embeddings = self.embed_action(actions.squeeze())+ time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go.float()) + time_embeddings
        
        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        #MADE a change (s1,a1,r1,s2,r2,a2....)
        h = torch.stack(
            (state_embeddings, action_embeddings_1,action_embeddings_2,returns_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 4 * T, self.h_dim)

        h = self.embed_ln(h)
        
        # transformer and prediction
        h = self.transformer(h)

        #print(
        #      torch.stack(
        #            (returns_embeddings, state_embeddings, action_embeddings), dim=1).shape,
        #      torch.stack(
        #                    (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #                ).permute(0, 2, 1, 3).shape,
        #    torch.stack(
        #                    (returns_embeddings, state_embeddings, action_embeddings), dim=1
        #                ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim).shape,
        #                      
        #    self.embed_ln(h).shape,
        #    h.shape
        #     )
        
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        
        h = h.reshape(B, T, 4, self.h_dim).permute(0, 2, 1, 3)
        ## get predictions
        #return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        #state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        ##action_preds = self.predict_action(h[:,1])  # predict action given r, s

        ##print(h.shape)

        #action_preds_1 = self.predict_actor_1(h[:,1])
        #action_preds_2 = self.predict_actor_2(h[:,1])
        #action_preds = torch.concat((action_preds_1,action_preds_2),axis =2)
    

        # MADE changes
        # h[:, 0, t] is conditioned on s_0, a_0 r_0, ... s_t
        # h[:, 1, t] is conditioned on s_0, a_0 r_0, ... s_t, a_t_1
        # h[:, 2, t] is conditioned on s_0, a_0 r_0, ... s_t, a_t_1,a_t_2
        # h[:, 3, t] is conditioned on s_0, a_0 r_0, ... s_t, a_t_1,a_t_2, r_t
        
        # get predictions
        return_preds = self.predict_rtg(h[:,0])     # predict next rtg given s, a,
        #return_preds = self.predict_rtg(h[:,3])     # predict next rtg given s, a,
        
        
        #state_preds = self.predict_state(h[:,1])    # predict next state given s, a #probably dont want this
        #action_preds = self.predict_action(h[:,1])  # predict action given r, s

        #print(h.shape)

        action_preds_1 = self.predict_actor_1(h[:,0])
        action_preds_2 = self.predict_actor_2(h[:,1]).clamp(min=0.01,max = 0.99)
        


        if return_logit == False: #returning actual values
            action_preds_1  =    nn.Tanh()(action_preds_1) if self.use_action_tanh else action_preds_1
            

        if return_log_prob_a2:
            
            dist = Beta(action_preds_2 * self.a2_concentration, (1 - action_preds_2) * self.a2_concentration)  # Adjust concentration parameters as needed
            logp_pi_a_2 = dist.log_prob(action_preds_2)
            
        elif return_og_log_prob_a2:
            dist = Beta(action_preds_2 * self.a2_concentration, (1 - action_preds_2) * self.a2_concentration)  # Adjust concentration parameters as needed
            logp_pi_a_2 = dist.log_prob(actions_2[:,:,None])
            #dist_entropy_a_2 = dist.entropy()
            #.clamp(min=0.0001)

            dist_entropy_a_1 = Categorical(nn.Softmax(dim=-1)(action_preds_1[:,-1,:])).entropy()


            
            return action_preds_1, action_preds_2, return_preds,logp_pi_a_2,dist_entropy_a_1#,dist_entropy_a_2
        else:
            logp_pi_a_2 = None
            

            #return action_preds_1, action_preds_2, return_preds


        
        #log_prob_2 = torch.log(action_preds_2) + torch.log(1 - action_preds_2)
        
        #action_preds = torch.concat((action_preds_1,action_preds_2),axis =2)
    

        #return state_preds, action_preds, return_preds
        return action_preds_1, action_preds_2, return_preds,logp_pi_a_2 #returning logits for action_1 and probability for action_2 
