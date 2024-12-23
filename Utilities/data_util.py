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



def get_dataset(env,episode_count=10,time_lim = 1000,model = None,learn_per_iter=10,random=10, reward_fn = (lambda terminated,truncated: -1 if terminated else (1 if truncated else 0) ),pkl_file_path='' ,cast = None, obs_in_info=False,predict_funct ='predict' , only_action=False,):

    print(obs_in_info)
    dataset = []

    data_ = collections.defaultdict(list)

    paths = []

    predict = getattr(model, predict_funct)
    
    for episode_step in tqdm.tqdm(range(episode_count),position=0):
        env.reset()
        observation, info = env.reset(seed=42)


        
        if obs_in_info:
            data_['observations'].append(info['observation'].astype(float))
        else:
            data_['observations'].append(observation)
        
        for tim in range(time_lim):
            #action = env.action_space.sample()
            if type(model)==type(None):
                action = env.action_space.sample()
            else:
                if np.random.rand()<random:
                    action, _states = predict(observation)
                    if cast != None:
                        action = cast(action)                    
                else:
                    action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            
            
            done_bool=  terminated or truncated or (tim == time_lim-1)
            #if terminated:
                #print(terminated)

            if obs_in_info:
                data_['next_observations'].append(info['observation'].astype(float))
            else:
                data_['next_observations'].append(observation)
            
            #data_['next_observations'].append(observation)
            data_['rewards'].append(reward_fn(terminated,truncated) )
            data_['terminals'].append(done_bool)
            data_['actions'].append([action])
            
            if done_bool:
                print(terminated , truncated , (tim == time_lim-1),tim)
                break
            else:
                if obs_in_info:
                    data_['observations'].append(info['observation'].astype(float))
                else:
                    data_['observations'].append(observation)
                #data_['observations'].append(observation)


        #print(episode_step)
        
                
        
        episode_data = {}
        for k in data_:
            
            #print(k,data_[k])
            episode_data[k] = np.array(data_[k])#,dtype=info['observation'].dtype)
        paths.append(episode_data)
        data_ = collections.defaultdict(list)
            

    env.close()

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    with open(f'{pkl_file_path}.pkl', 'wb') as f:
        pickle.dump(paths, f)
    
    return paths
    