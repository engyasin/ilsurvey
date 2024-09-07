

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class PPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs[0].single_observation_space).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs[0].single_observation_space).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs[0].single_action_space[0]), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)



class PPOAgentwithZ(nn.Module):
    def __init__(self, envs, z_classes=3):
        super().__init__()

        self.z_classes = z_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs[0].single_observation_space).prod()+self.z_classes, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs[0].single_observation_space).prod()+self.z_classes, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs[0].single_action_space[0]), std=0.01),
        )

    def get_value(self, x, z=None):

        if z is None:
            z = torch.randint(0,3,(x.shape[0],))
            z = torch.functional.F.one_hot(z,num_classes=self.z_classes).float().to(self.device)

        x = torch.hstack((x,z))

        return self.critic(x)

    def get_action_and_value(self, x, action=None, z=None):

        if z is None:
            z = torch.randint(0,3,(x.shape[0],))
            z = torch.functional.F.one_hot(z,num_classes=self.z_classes).float().to(self.device)
        x = torch.hstack((x,z))

        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


    

# ALGO LOGIC: initialize agent here:
class DQNAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(env[0].single_observation_space).prod(), 120)),
            nn.ReLU(),
            layer_init(nn.Linear(120, 84)),
            nn.ReLU(),
            layer_init(nn.Linear(84, env[0].single_action_space[0])),
        )

    def forward(self, x):
        return self.network(x)
    



# PHI network for maxnet

class phi(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.phinet= nn.Sequential(
            layer_init(nn.Linear(np.array(env[0].single_observation_space).prod(), 64),std=8),
            nn.ReLU(),
            layer_init(nn.Linear(64, 4),std=16,bias_const=5.0),
            nn.ReLU(),
            #layer_init(nn.Linear(32, 4)),
            #nn.ReLU(),
        )

    def forward(self, x):
        x  = self.phinet(x)
        #distribution = Normal(x[:,:4],x[:,4:]+1e-5)
        #return distribution.rsample()
        return x
    
    def get_v(self,state,action=None):

        allphis = self.phinet(state)
        return -torch.linalg.norm((allphis[:,:2]-allphis[:,2:]),axis=1)
    



class Discriminator(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(
                np.array(envs[0].single_observation_space).prod()+envs[0].single_action_space[0],64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=8),
        )

    def forward(self, state,action):
        x = torch.hstack((state,action))
        x  = self.net(x)
        return x
    


class Discriminator_AIRL(nn.Module):
    def __init__(self, envs,agent=None):
        super().__init__()

        self.net_fw = nn.Sequential(
            layer_init(nn.Linear(
                np.array(envs[0].single_observation_space).prod()+envs[0].single_action_space[0],64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=8),
            #nn.ReLU(),
        )

        self.agent = agent

    def forward(self, state,action):

        return torch.logit(self.get_prob(state,action))
    
    def get_prob(self,state,action):

        x = torch.hstack((state,action))
        #x  = torch.exp(self.net_fw(x))

        with torch.no_grad():
            _,logprobx, _ ,_  = self.agent.get_action_and_value(state, action=action.argmax(axis=1))
        #if action.shape[0]>1:
        #    breakpoint()
        return 1/(1+torch.exp(logprobx[:,None]-self.net_fw(x)))
        #return (x/(x+torch.exp(logprobx[:,None])))



class Qnet(nn.Module):
    def __init__(self, envs, classes=3):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(
                np.array(envs[0].single_observation_space).prod()+envs[0].single_action_space[0],64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, classes), std=2),
            nn.Softmax()
        )

    def forward(self, state,action):
        x = torch.hstack((state,action))
        x  = self.net(x)
        return x
    