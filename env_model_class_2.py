#env = gym.make(env_name,max_episode_steps=200)

import gymnasium as gym
import numpy as np
from typing import Optional
import utils_gym


env = gym.make('MountainCar-v0',max_episode_steps=2000)

class model_class_2(type(env)):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.state_list = np.zeros((3,2))
        self.observation_space = gym.spaces.Box(low=-10 ,high=20, shape=(6,), dtype=np.float32)

    def step(self, action):
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        self.state_list[:2,:] = self.state_list[1:,:]
        self.state_list[2,:] = observation
        info['observation'] = observation.astype(np.float32)

        return self.state_list.reshape(-1), reward, terminated, truncated, info
        
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            default = False,
            
            init_state=[-5,0]
                    ):
            super().reset(seed=seed)
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            if default:
                low, high = utils_gym.maybe_parse_reset_bounds(options, -0.6, -0.4)
                self.state = np.array([self.np_random.uniform(low=low, high=high), 0])
            else:
                self.state = np.array(init_state,dtype='float32')
    
            if self.render_mode == "human":
                self.render()

            self.state_list = np.array([self.state],dtype=np.float32).repeat(3,axis =0)
            
            return self.state_list.reshape(-1), {'observation':self.state}
        


def make_env(env_id, idx, capture_video, run_name,custom_env=False,max_episode_steps=2000):
    def thunk():

                    
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
                
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:

            env = gym.make(env_id)

            
            #env = gym.make(env_id)
        if custom_env:
            #print('here')
            env = model_class_2(env=env,max_episode_steps=max_episode_steps)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


