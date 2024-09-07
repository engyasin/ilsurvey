import numpy as np
import cv2

from doorsenvs import Doors,DoorZ


from agents import PPOAgent
import torch
import torch.optim as optim


def collect_expert_trajs():

    traj_len = 32
    env = Doors(max_steps=traj_len)

    x = 0
    obs = env.reset(agent_at=x)
    n_episodes = 15
    all_actions = [[]]
    all_obses = [[]]
    all_features = []
    for i in range(traj_len*n_episodes):

        all_obses[-1].append(obs.copy())
        all_features.append(np.hstack((env.goal,env.agent)))
        a = env.expert_action()
        env.render(scale=20)
        obs,reward,done,info = env.step(a)

        cv2.waitKey(10)
        all_actions[-1].append(a)


        if done:
            x+=1
            obs = env.reset(agent_at=min(x,n_episodes-1))
            all_actions.append([])
            all_obses.append([])

    print(np.array(all_actions[:-1]).shape)
    print(np.array(all_obses[:-1]).shape)
    np.save(f'expert_actions_15_{traj_len}.npy',np.array(all_actions[:-1]))
    np.save(f'expert_obses_15_{traj_len}_225.npy',np.array(all_obses[:-1]))
    np.save(f'expert_features_15_{traj_len}.npy',np.array(all_features[:]))
    env.close()


def collect_expert_trajs_modes():

    traj_len = 32
    env = DoorZ(max_steps=traj_len)

    x = 0
    obs = env.reset(agent_at=x)
    n_episodes = 15*3
    all_actions = [[]]
    all_obses = [[]]
    all_features = []
    for i in range(traj_len*n_episodes):

        all_obses[-1].append(obs.copy())
        all_features.append(np.hstack((env.goal,env.agent)))
        a = env.expert_action()
        env.render(scale=20)
        obs,reward,done,info = env.step(a)

        cv2.waitKey(10)
        all_actions[-1].append(a)


        if done:
            x+=1
            obs = env.reset(agent_at=x%env.gridsize[1])#min(x,n_episodes-1))
            all_actions.append([])
            all_obses.append([])

    print(np.array(all_actions[:-1]).shape)
    print(np.array(all_obses[:-1]).shape)
    np.save(f'expert_actions_15_{traj_len}_Z.npy',np.array(all_actions[:-1]))
    np.save(f'expert_obses_15_{traj_len}_225_Z.npy',np.array(all_obses[:-1]))
    np.save(f'expert_features_15_{traj_len}_Z.npy',np.array(all_features[:]))
    env.close()




def main():
    #collect_expert_trajs()
    traj_len = 32
    training_epochs = 3000
    lr = 9e-4
    demos_size = 240
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Doors(max_steps=traj_len)
    agent = PPOAgent(envs=[env]).to(device=device).float()
    optimizer = optim.Adam(agent.actor.parameters(), lr=lr, eps=1e-5)

    expert_actions = torch.as_tensor(np.load(f'expert_actions_15_{traj_len}.npy').flatten(),device=device).long()[:demos_size]
    onehot_expert_acts = torch.functional.F.one_hot(expert_actions,num_classes=5).float()
    expert_obs = torch.as_tensor(np.load(f'expert_obses_15_{traj_len}_225.npy').reshape(-1,np.prod(env.gridsize)),device=device).float()[:demos_size,:]


    agent.actor.train()

    for epoch in range(training_epochs):
        out = agent.actor(expert_obs)
        loss = torch.functional.F.binary_cross_entropy_with_logits(out,onehot_expert_acts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

    torch.save(agent,f'bc_agent_iter_{training_epochs}_mlp.pth')


if __name__=='__main__':

    collect_expert_trajs_modes()