

import numpy as np
import cv2

from agents import PPOAgent as Agent
from agents import Qnet,PPOAgentwithZ, Discriminator_AIRL

import torch

import imageio

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')


class Doors():

    def __init__(self,gridsize=[15,15],doors=3,max_steps= 32,seed=42) -> None:
        
        
        # grid edge is always odd
        self.gridsize = gridsize
        self.doors = doors
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed(seed=seed)
        self.reset()
        self.single_observation_space = (np.prod(gridsize),)
        self.single_action_space = (len(self.actions_dict),)

        self.max_steps = max_steps

        self.discriminator = None
        self.qnet = None
        self.current_z = 0

    def step(self,action):

        action_rc = self.actions_dict[int(action)]

        #evacuate if not initial
        if self.time:
            self.grid[self.agent[0],self.agent[1]] = 0

        # stop at edges
        agent_rc =  np.clip(self.agent + action_rc,[0,0],np.array(self.gridsize)-1)
        self.prv_agent = self.agent.copy()
        # stop at walls
        if self.grid[agent_rc[0],agent_rc[1]]==0:
            self.agent = agent_rc

        self.grid[self.agent[0],self.agent[1]] = 1

        reward = self.reward(self.grid,action)
        self.episodic_return += self.gt_reward()#reward
        self.time += 1

        done = (self.time>=self.max_steps)# or (reward==0)

        state = self.grid.flatten().copy()
        info = {}
        if done:
            info.update({'episode':{'r':self.episodic_return,'l':self.time}})
            info.update({"terminal_observation":state.copy()})
            state = self.reset()

        return state, reward, done, info


    def gt_reward(self):

        v0 =  -np.sqrt(((self.prv_agent-self.goal)**2).sum())
        v1 =  -np.sqrt(((self.agent-self.goal)**2).sum())

        return v1-v0



    def reward(self,state,action):

        if  self.discriminator :
            with torch.no_grad():
                state = torch.as_tensor((state)).to(device=self.device).flatten().float()[None,:]
                action = torch.as_tensor((action)).to(device=self.device)
                action = torch.functional.F.one_hot(action,num_classes=5).float()[None,:]
                logits = self.discriminator(state,action)[0]
                #r=0
                #if self.qnet:
                #    r += torch.functional.F.logsigmoid(self.qnet(state,action)[self.current_z]).detach().cpu().numpy()

            return torch.functional.F.logsigmoid(logits).detach().cpu().numpy()#logits.detach().cpu().numpy()#



        return self.gt_reward()

    def seed(self,seed=42):
        np.random.seed(seed)

    def expert_action(self):

        move_vector = (self.goal-self.agent).astype(float)
        len_move_vec = np.linalg.norm(move_vector)

        if len_move_vec:

            if self.agent[0]>(self.gridsize[0]//2):
                dist_2_goals = np.linalg.norm((np.array(self.door_blocks)-self.agent).astype(float),axis=1)
                newgoal = self.door_blocks[dist_2_goals.argmin()]# smallest distance
                move_vector = (newgoal-self.agent).astype(float)
                len_move_vec = np.linalg.norm(move_vector)

            move_vector /= len_move_vec

            actions_dists = np.linalg.norm(self.actions_dict - move_vector,axis=1)
            actions_dists[0] = 1000 # always move
            best_action = actions_dists.argmin()
            while (self.agent+self.actions_dict[best_action]).tolist() in self.wall_blocks:
                actions_dists[best_action] = 1000
                best_action = actions_dists.argmin()
        else:
            best_action = 0
        
        return best_action

    def reset(self,agent_at=-1):

        self.grid = np.zeros(self.gridsize)

        self.actions_dict = np.array([[0,0],[0,1],[1,0],[0,-1],[-1,0]]).astype(int)

        self.grid[-self.gridsize[0]//2] = -1

        self.door_blocks = []
        for d in range(self.doors):
            door_col = (self.gridsize[1]//self.doors)*(d+1)
            self.grid[-self.gridsize[0]//2][door_col-1] = 0
            self.door_blocks.append([(self.gridsize[0]//2),door_col-1])

        self.wall_blocks = np.vstack(np.where(self.grid)).T.tolist()

        self.agent = np.array([self.gridsize[0]-1,np.random.randint(0,self.gridsize[1])]).astype(int) #r,c
        if agent_at>=0:
            self.agent[1] = agent_at
        self.grid[self.agent[0],self.agent[1]] = 2

        self.initial_location = self.agent.copy()

        self.goal = np.array([0,self.gridsize[1]-1-self.agent[1]])

        self.time = 0
        self.episodic_return = 0

        self.prv_agent = self.agent.copy()

        return self.grid.flatten()


    def get_feature(self):

        # dynamic
        # agent location, initial location, distance between them ,time step
        dist = np.linalg.norm(self.initial_location-self.agent)

        return np.append(self.initial_location,self.agent,np.array([dist,self.time]))

    def render(self,scale=10,return_image=False):

        img = (np.dstack([self.grid==2,self.grid==-1,self.grid==1])*1.0)#.astype(np.uint16)
        img = img.repeat(scale,axis=0).repeat(scale,axis=1)
        cv2.imshow('Doors',img)

        if return_image: return img

    def close(self):

        cv2.destroyAllWindows()



class DoorZ(Doors):

    def __init__(self, gridsize=[15, 15], doors=3, max_steps=32, seed=42, qnet=None) -> None:

        self.classes = 3

        self.qnet = qnet
        super().__init__(gridsize, doors, max_steps, seed)


    def reward(self,state,action):

        if  self.discriminator :
            with torch.no_grad():
                state = torch.as_tensor((state)).to(device=self.device).flatten().float()[None,:]
                action = torch.as_tensor((action)).to(device=self.device)
                action = torch.functional.F.one_hot(action,num_classes=5).float()[None,:]
                logits = self.discriminator(state,action)[0]
                r=0
                if self.qnet:
                    r += torch.log(self.qnet(state,action)[:,self.gt_z.argmax().item()]).detach().cpu().numpy()*3
            return r+torch.functional.F.logsigmoid(logits).detach().cpu().numpy()#logits.detach().cpu().numpy()#
        
    def reset(self, agent_at=-1):
        out =  super().reset(agent_at)

        self.gt_z = torch.functional.F.one_hot(
            torch.as_tensor(self.goal[1]//(self.gridsize[1]//self.classes)),
            num_classes=self.classes).to(device=self.device).float()

        z = torch.randint(0,3,(1,))
        self.gt_z = torch.functional.F.one_hot(z,num_classes=self.classes).float().to(self.device)

        return out

    def expert_action(self):

        move_vector = (self.goal-self.agent).astype(float)
        len_move_vec = np.linalg.norm(move_vector)

        if len_move_vec:

            if self.agent[0]>(self.gridsize[0]//2):
                newgoal = self.door_blocks[self.gt_z.argmax().item()]# smallest distance
                move_vector = (newgoal-self.agent).astype(float)
                len_move_vec = np.linalg.norm(move_vector)

            move_vector /= len_move_vec

            actions_dists = np.linalg.norm(self.actions_dict - move_vector,axis=1)
            actions_dists[0] = 1000 # always move
            best_action = actions_dists.argmin()
            while (self.agent+self.actions_dict[best_action]).tolist() in self.wall_blocks:
                actions_dists[best_action] = 1000
                best_action = actions_dists.argmin()
        else:
            best_action = 0
        
        return best_action

def main():
    
    env = Doors(max_steps=32)

    alg = 7

    if alg == 0:
        agent = torch.load(f'ppo_agent_iter_{1000000}_mlp.pth')
    elif alg == 1:
        agent = torch.load(f'dqn_qnet_iter_{1000000}_mlp.pth')
    elif alg == 2:
        agent = torch.load(f'bc_agent_iter_{3000}_mlp.pth')
    elif alg == 3:
        agent = torch.load(f'Reward_maxnet_agent_iter_{5000}_mlp.pth')
    elif alg == 4:
        agent = torch.load(f'gail_agent_iter_{160}_mlp.pth')
    elif alg == 5:
        agent = torch.load(f'airl_agent_iter_{160}_mlp.pth')
        fw_net = torch.load(f'airl_reward_iter_{160}_mlp.pth')
    elif alg ==6:
        agent = torch.load(f'infogail_agent_iter_{80}_mlp.pth')
        qnet = torch.load(f'infogail_qnet_iter_{80}_mlp.pth')


    obs = env.reset(agent_at=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_episodes = 15
    all_returns = []
    all_rewards = []
    all_imgs = []
    z = 0
    for aa in np.arange(32*n_episodes):

        if alg in [0,2,3,4,5]:
            a = agent.actor(torch.as_tensor(obs[None,:],device=device).float()).argmax(dim=1)
        elif alg == 1:
            a = agent(torch.as_tensor(obs[None,:],device=device).float()).argmax(dim=1) #dqn
        elif alg == 6:
            obs_ = torch.as_tensor(obs[None,:],device=device).float()
            a = agent.actor(torch.hstack((obs_,torch.as_tensor([[0,0,1]]).float().to(device))))
            z += qnet(obs_,a)
            a =  a.argmax(dim=1)
        else:
            a = env.expert_action()

        obs,reward,done,info = env.step(a)

        all_imgs.append(env.render(scale=20,return_image=True).astype(np.uint8)[:,:,::-1]*255)
        #all_rewards.append([(fw_net.net_fw(torch.as_tensor(np.hstack((obs,np.eye(5)[a])),device=device).float()).item()),reward])
        cv2.waitKey(10)
        if done:
            all_returns.append(1*info['episode']['r'])
            obs = env.reset(agent_at=((aa)//31)%15)

    #plt.plot(all_returns)
    #print(z/z.sum())
    #plt.show()
    print(all_returns)
    print(np.std(all_returns))
    print(np.mean(all_returns))
    env.close()
    #imageio.mimsave(f'images/gifs/expert.gif', all_imgs, 'GIF')
    if False:
        all_rewards = np.array(all_rewards)
        all_rewards[:,0] -= all_rewards[:,0].min()
        all_rewards[:,0] /= all_rewards[:,0].max()
        all_rewards[:,0] = (all_rewards[:,0]*2)-1
        plt.plot((all_rewards)[:,0],label='AIRL advantage')
        plt.plot((all_rewards)[:,1], label='Ground Truth reward')

        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Advantage function')
        plt.show()

# Expert         : Returns: 16.33 STD: 1.97
# PPO (GT reward): Returns: 15.26 STD: 3.03
# DQN (GT reward): Returns: 15.15 STD: 4.42

# half the expert set


# BC:     Returns: 11.49 STD: 5.34
# MAXENT: Returns: 11.56 STD: 5.32
# GAIL:   Returns: 12.07 STD: 3.42 (more is possible with more training)
# AIRL:   Returns: 11.5 STD: 4.27 (more is possible with more training)


if __name__=='__main__':

    main()
