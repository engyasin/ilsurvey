import numpy as np
import cv2

from doorsenvs import Doors


from agents import phi,PPOAgent
import torch
import torch.optim as optim

from matplotlib import pyplot as plt
from torch.distributions.categorical import Categorical


def findR(phi_array):

    return np.exp(-np.linalg.norm((phi_array[:,:2]-phi_array[:,2:4]),axis=0))


def returnAllStates(templateState):

    empty_state = np.clip(-1,0,templateState)

    all_states = []
    agent_states = []
    gt_phis = []
    for row in range(empty_state.shape[0]):
        for place in range(empty_state.shape[1]):
            if empty_state[row][place] == 0:
                agent_states.append(empty_state.copy())
                agent_states[-1][row][place] = 1
                gt_phis.append([row,place])

    agent_based_states = np.dstack(agent_states)

    gt_phis = np.array(gt_phis)
    all_gt_phis = []

    for j in range(empty_state.shape[1]):

        all_states.append(agent_based_states.copy())
        all_states[-1][empty_state.shape[0]-1,j,:] = 2
        all_gt_phis.append(np.hstack((gt_phis.copy(),
                                      np.array([[[empty_state.shape[0]-1,j]]*gt_phis.shape[0]])[0])))

    all_gt_phis = np.vstack(all_gt_phis)

    # 15*15*10 = 2250 state
    return np.concatenate(all_states,axis=2),all_gt_phis


def main2():
    # NOTE: assumming we know it's distance function

    traj_len = 32
    training_epochs = 5000
    demos_size = 240
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Doors(max_steps=traj_len)
    Ragent = PPOAgent(envs=[env]).to(device=device).float()
    optimizer = optim.Adam(Ragent.actor.parameters(), lr=lr, eps=1e-5)

    with torch.no_grad():
        expert_actions = torch.as_tensor(np.load(f'expert_actions_15_{traj_len}.npy').flatten(),device=device).long()[:demos_size]
        #onehot_expert_acts = torch.functional.F.one_hot(expert_actions,num_classes=5).float()
        expert_obs = torch.as_tensor(np.load(f'expert_obses_15_{traj_len}_225.npy').reshape(-1,np.prod(env.gridsize)),device=device).float()[:demos_size,:]
        all_states,all_gt_phis = returnAllStates(env.grid)
        allStates = torch.as_tensor(all_states,device=device).float().reshape(np.prod(env.grid.shape),-1).T

    Ragent.train()


    for epoch in range(training_epochs):
        _,logprob_all,entropy,_ = Ragent.get_action_and_value(expert_obs,action=expert_actions)
        logprob = logprob_all.mean()
        #logits = Ragent.actor(allStates)
        #entropy = Categorical(logits=logits).entropy()

        loss = -(logprob*1 + entropy.mean()*0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%(200)==0:
            print(logprob.item(),entropy.mean().item())


    torch.save(Ragent,f'Reward_maxnet_agent_iter_{training_epochs}_mlp.pth')



def main():

    # NOTE: assumming we know it's distance function

    traj_len = 32
    training_epochs = 5000
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Doors(max_steps=traj_len)
    #agent = PPOAgent(envs=[env]).to(device=device).float()
    phinet = phi(env=[env]).to(device=device).float()
    optimizer = optim.Adam(phinet.parameters(), lr=lr, eps=1e-5)

    with torch.no_grad():
        #expert_actions = torch.as_tensor(np.load(f'expert_actions_15_{traj_len}.npy').flatten(),device=device).long()
        #onehot_expert_acts = torch.functional.F.one_hot(expert_actions,num_classes=5).float()
        expert_obs = torch.as_tensor(np.load(f'expert_obses_15_{traj_len}_225.npy').reshape(-1,np.prod(env.gridsize)),device=device).float()
        phi_obs = torch.as_tensor(np.load(f'expert_features_15_{traj_len}.npy'),device=device).float().mean(axis=0)
        all_states,all_gt_phis = returnAllStates(env.grid)
        allStates = torch.as_tensor(all_states,device=device).float().reshape(np.prod(env.grid.shape),-1).T

    distances_gt = np.linalg.norm((all_gt_phis[:,:2]-all_gt_phis[:,2:4]),axis=1)
    phinet.train()
    for epoch in range(training_epochs):

        # find phi_obs
        #phi_obs = phinet(expert_obs).mean(axis=0)

        # find Z 
        allphis = phinet(allStates)
        all_ps = torch.exp(-torch.linalg.norm((allphis[:,:2]-allphis[:,2:4]),axis=1))

        # Z (item)
        Z = all_ps.sum()#.item()
        estimated_phi_s = ((all_ps * allphis.T)/Z).sum(axis=1)
        loss = (abs(phi_obs - estimated_phi_s)**2).sum()# + torch.exp(-(estimated_phi_s.sum()))*1
        #l = 0
        #for param in phinet.parameters():
        #    l+=(param).mean()

        #loss += torch.exp(-l)*50
        #loss -= l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

        distances_hat = torch.linalg.norm((allphis[:,:2]-allphis[:,2:4]),axis=1).detach().cpu().numpy()
        diff_ = abs((distances_hat*(19.8/distances_hat.max()))-(distances_gt))
        print(diff_.mean(),diff_.std())

    #breakpoint()

    # test

    phinet.eval()
    with torch.no_grad():
        allphis = phinet(allStates).cpu().numpy()

    #distances_hat = np.linalg.norm((allphis[:,:2]-allphis[:,2:]),axis=1)
    #distances_gt = np.linalg.norm((all_gt_phis[:,:2]-all_gt_phis[:,2:]),axis=1)
    #breakpoint()
    torch.save(phinet,f'phi_agent_iter_{training_epochs}_mlp.pth')

    # TODO draw heatmaps



if __name__ == '__main__':

    main2()