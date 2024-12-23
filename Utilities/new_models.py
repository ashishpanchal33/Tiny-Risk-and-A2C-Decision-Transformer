# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from typing import Optional
@dataclass
class Args:
    exp_name: str = 'exp1_A2C_DT_'#os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    TB_log:bool = True
    custom_env = True

    # Algorithm specific arguments
    env_id: str = "Tiny-Risk"#"MountainCar-v0"#"CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4#0.001#0.001 #2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1#4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999 #0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.98 #0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 20#4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""


    learning_starts: int = 25e3
    tau: float = 0.005
    buffer_size: int = 60000
    # to be filled in runtime
    batch_size: int = 256
    """the batch size (computed in runtime)"""
    minibatch_size: int = 8
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 10
    """the number of iterations (computed in runtime)"""

#gae_lambda": 0.98, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "batch_size": 64, "n_epochs": 4


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self,envs,ob_space=None):
        if not ob_space:
            ob_space = envs.single_observation_space.shape
        
        
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(ob_space).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(ob_space).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    def predict(self, x, action=None):
        return get_action_and_value(self, x, action=None)[0]



class Agent_shared(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    def predict(self, x, action=None):
        return get_action_and_value(self, x, action=None)[0]



class Agent_shared_v1(nn.Module):
    def __init__(self, action_space = None, ob_space=None):
        super(Agent_shared_v1, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(ob_space, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.GELU(),
        )
        self.actor = layer_init(nn.Linear(64, action_space), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def predict(self, x, action=None):
        return get_action_and_value(self, x, action=None)[0]
    def get_valid_action(self,x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs















class Agent_shared_v2(nn.Module):
    def __init__(self, action_space = None, ob_space=None):
                
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(ob_space, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(ob_space, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_space), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    def predict(self, x, action=None):
        return get_action_and_value(self, x, action=None)[0]





class Agent_shared_v1_risk(nn.Module):
    def __init__(self, action_space = None, ob_space=None):
        super(Agent_shared_v1_risk, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(ob_space, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.GELU(),
        )
        self.actor_1 = layer_init(nn.Linear(64, action_space), std=0.01)
        self.actor_2 = nn.Sequential(layer_init(nn.Linear(64, 2), std=0.01), nn.Softmax(dim=1),)
        
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None,training=False):
        hidden = self.network(x)
        logits = self.actor_1(hidden)
        probs = Categorical(logits=logits)

        
        if action is None:
            action_1 = probs.sample()[:,None]
            action_2 = self.actor_2(hidden)[:,[0]]
    
            action = torch.concat((action_1,action_2),1)
        else:
            action_1,action_2 = action[:,0],action[:,1]

        if (len(x)>1) or (training):
            ret_p = probs.log_prob(action_1)*action_2
        else:
            ret_p = probs.log_prob(action_1[:,0])*action_2[:,0]

        
        
        return action, ret_p, probs.entropy(), self.critic(hidden)

    def predict(self, x, action=None):
        return get_action_and_value(self, x, action=None)[0]
    def get_valid_action(self,x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        action_2 = self.actor_2(hidden)[:,[0]]
        
        return probs.probs,action_2





# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):

    def __init__(self, action_space = None, ob_space=None):
        super().__init__()
        self.networkx = nn.Sequential(
            layer_init(nn.Linear(ob_space+action_space, 256)),
            nn.GELU(),
            #layer_init(nn.Linear(256, 256)),
            #nn.GELU(),
            layer_init(nn.Linear(256, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1)),
            nn.GELU(),
        )
        #self.actor_1 = layer_init(nn.Linear(64, action_space), std=0.01)
        #self.actor_2 = nn.Sequential(layer_init(nn.Linear(64, 2), std=0.01), nn.Softmax(dim=1),)
    def forward(self, x,a):
        x = torch.cat([x, a], 1)
        val = self.networkx(x)
        #hidden = self.network(x)
        #logits = self.actor_1(hidden)
        #probs = Categorical(logits=logits)

        #action_1 = probs.sample()[:,None]
        #action_2 = self.actor_2(hidden)[:,[0]]

        #action = torch.concat((action_1,action_2),1)

        return val #action


# ALGO LOGIC: initialize agent here:
class QNetwork_small(nn.Module):

    def __init__(self, action_space = None, ob_space=None):
        super().__init__()
        self.networkx = nn.Sequential(
            layer_init(nn.Linear(ob_space+action_space, 64)),
            nn.GELU(),
            #layer_init(nn.Linear(256, 256)),
            #nn.GELU(),
            layer_init(nn.Linear(64, 32)),
            nn.GELU(),
            layer_init(nn.Linear(32, 1)),
            nn.GELU(),
        )
        #self.actor_1 = layer_init(nn.Linear(64, action_space), std=0.01)
        #self.actor_2 = nn.Sequential(layer_init(nn.Linear(64, 2), std=0.01), nn.Softmax(dim=1),)
    def forward(self, x,a):
        x = torch.cat([x, a], 1)
        val = self.networkx(x)
        #hidden = self.network(x)
        #logits = self.actor_1(hidden)
        #probs = Categorical(logits=logits)

        #action_1 = probs.sample()[:,None]
        #action_2 = self.actor_2(hidden)[:,[0]]

        #action = torch.concat((action_1,action_2),1)

        return val #action




class QNetwork_duelingdqn(nn.Module):
    def __init__(self, action_space = None, ob_space=None):
        super().__init__()
        self.networkx = nn.Sequential(
            layer_init(nn.Linear(ob_space, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            #layer_init(nn.Linear(64, 64)),
            #nn.GELU(),
        )
        self.value = layer_init(nn.Linear(64, 1), std=0.01)
        self.advantage = layer_init(nn.Linear(64+action_space, 1), std=0.01)
        
    def forward(self, x,action):
        
        hidden = self.networkx(x).reshape(-1,64)
        val = self.value(hidden)

        x_action = torch.concat((hidden,action),1)
        advantage = self.advantage(x_action)

        return val+advantage #,val, advantage #* self.action_scale + self.action_bias



class Actor_ddqn(nn.Module):
    def __init__(self,env, action_space = None, ob_space=None):
        super().__init__()
        #self.fc1 = nn.Linear(ob_space, 256)
        #self.fc2 = nn.Linear(256, 256)
        #self.fc_mu = nn.Linear(256, action_space)


        self.network = nn.Sequential(
            layer_init(nn.Linear(ob_space, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            #layer_init(nn.Linear(64, 64)),
            #nn.GELU(),
        )
        self.actor_1 = layer_init(nn.Linear(64, action_space), std=0.01)
        self.actor_2 = nn.Sequential(layer_init(nn.Linear(64, 2), std=0.01), nn.Softmax(dim=1),)



        
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space(1).high - env.action_space(1).low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space(1).high + env.action_space(1).low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, return_prob=False):
        
        hidden = self.network(x)
        logits = self.actor_1(hidden)
        probs = Categorical(logits=logits)

        action_1 = probs.sample()[:,None]
        action_2 = self.actor_2(hidden)[:,[0]]

        action = torch.concat((action_1,action_2),1)
        if return_prob==1:
            probs_ = probs.probs*action_2
            
            return action, probs_
        elif return_prob ==2:
            return action, probs.probs
        else:
            return action#* self.action_scale + self.action_bias






class QNetwork_duelingdqn_small(nn.Module):
    def __init__(self, action_space = None, ob_space=None):
        super().__init__()
        self.networkx = nn.Sequential(
            layer_init(nn.Linear(ob_space, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            layer_init(nn.Linear(64, 32)),
            nn.LayerNorm(32),
            nn.GELU(),
            #layer_init(nn.Linear(64, 64)),
            #nn.GELU(),
        )
        self.value = layer_init(nn.Linear(32, 1), std=0.01)
        self.advantage = layer_init(nn.Linear(32+action_space, 1), std=0.01)
        
    def forward(self, x,action):
        
        hidden = self.networkx(x).reshape(-1,32)
        val = self.value(hidden)

        x_action = torch.concat((hidden,action),1)
        advantage = self.advantage(x_action)

        return val+advantage #,val, advantage #* self.action_scale + self.action_bias



class Actor_ddqn_small(nn.Module):
    def __init__(self,env, action_space = None, ob_space=None):
        super().__init__()
        #self.fc1 = nn.Linear(ob_space, 256)
        #self.fc2 = nn.Linear(256, 256)
        #self.fc_mu = nn.Linear(256, action_space)


        self.network = nn.Sequential(
            layer_init(nn.Linear(ob_space, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            layer_init(nn.Linear(64, 32)),
            nn.LayerNorm(32),
            nn.GELU(),
            #layer_init(nn.Linear(64, 64)),
            #nn.GELU(),
        )
        self.actor_1 = layer_init(nn.Linear(32, action_space), std=0.01)
        self.actor_2 = nn.Sequential(layer_init(nn.Linear(32, 2), std=0.01), nn.Softmax(dim=1),)



        
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space(1).high - env.action_space(1).low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space(1).high + env.action_space(1).low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, return_prob=False):
        
        hidden = self.network(x)
        logits = self.actor_1(hidden)
        probs = Categorical(logits=logits)

        action_1 = probs.sample()[:,None]
        action_2 = self.actor_2(hidden)[:,[0]]

        action = torch.concat((action_1,action_2),1)
        if return_prob==1:
            probs_ = probs.probs*action_2
            
            return action, probs_
        elif return_prob ==2:
            return action, probs.probs
        else:
            return action#* self.action_scale + self.action_bias


    
#    def forward(self, x):
#        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)






















