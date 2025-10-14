import numpy as np
import torch

class DataObj:

    def __init__(self,observations,next_observations,actions,rewards,dones):

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones




class ReplayBuffer:


    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=4, handle_timeout_termination=True):

        self.buffer_size = buffer_size * n_envs

        self.observations = np.array([observation_space.sample()]*n_envs)
        self.next_observations = np.array([observation_space.sample()]*n_envs)
        self.actions = np.array([action_space.sample()]*n_envs)[:,None]
        self.rewards = np.ones((n_envs,1))#[None,...]
        self.dones = np.ones((n_envs,1))#[None,...]

        self.device = device


    def add(self, obs, real_next_obs,  actions, 
                   rewards, dones):

        self.observations = np.vstack((self.observations,obs))[-self.buffer_size:,...]
        self.next_observations = np.vstack((self.next_observations,real_next_obs))[-self.buffer_size:,...]
        self.actions = np.vstack((self.actions,actions[:,None]))[-self.buffer_size:,...]
        self.rewards = np.vstack((self.rewards,rewards[:,None]))[-self.buffer_size:,...]
        self.dones = np.vstack((self.dones,dones[:,None]))[-self.buffer_size:,...]

    def sample(self,size=1):

        indx = np.arange(self.observations.shape[0]-1)
        np.random.shuffle(indx)

        samples_indx = indx[:size]+1

        return DataObj(torch.Tensor(self.observations[samples_indx]).to(self.device),
                       torch.Tensor(self.next_observations[samples_indx]).to(self.device),
                       torch.Tensor(self.actions[samples_indx]).to(self.device),
                       torch.Tensor(self.rewards[samples_indx]).to(self.device),
                       torch.Tensor(self.dones[samples_indx]).to(self.device))
