
import numpy as np
import cv2

from doorsenvs import Doors


from agents import PPOAgent,Discriminator
import torch
import torch.optim as optim

from ppo_class import ppo_trainer



def main():

    traj_len = 32
    training_epochs = 160
    policy_epochs = 10000
    discriminator_epochs = 16
    demo_size = 240

    lr = 3e-5
    lr_d = 2e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Doors(max_steps=traj_len)

    discriminator = Discriminator(envs=[env]).to(device=device).float()
    agent = PPOAgent(envs=[env]).to(device=device).float()
    #optimizer = optim.Adam(agent.actor.parameters(), lr=lr, eps=1e-5)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, eps=1e-5)

    expert_actions = torch.as_tensor(np.load(f'expert_actions_15_{traj_len}.npy').flatten(),device=device).long()[:demo_size]
    onehot_expert_acts = torch.functional.F.one_hot(expert_actions,num_classes=5).float()
    expert_obs = torch.as_tensor(np.load(f'expert_obses_15_{traj_len}_225.npy').reshape(-1,np.prod(env.gridsize)),device=device).float()[:demo_size,:]


    agent.actor.train()
    discriminator.train()
    env.discriminator = discriminator

    # TODO show the real reward
    trainer = ppo_trainer(agent=agent, lr=lr, total_timesteps=policy_epochs, buffer_size=expert_obs.shape[0],
                          env_id='GAIL')
    # different envs
    for env_i,_ in enumerate(trainer.envs):
        trainer.envs[env_i].discriminator = discriminator

    for epoch in range(training_epochs):

        # policy training

        generator_obs, generator_acts, _ = trainer.train()
        # discriminator training

        training_batch_obs = torch.vstack((expert_obs,generator_obs))
        training_batch_acts = torch.vstack((onehot_expert_acts,generator_acts))

        #training_batch = torch.hstack((training_batch_obs,training_batch_acts))

        labels = torch.vstack((torch.ones_like(expert_obs[:,:1]),torch.zeros_like(generator_obs[:,:1])))


        for epoch_d in range(discriminator_epochs):
            # k

            out = discriminator(training_batch_obs,training_batch_acts)
            loss = torch.functional.F.binary_cross_entropy_with_logits(out,labels)
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()

            
            #print(loss.item())
        classification = (torch.functional.F.sigmoid(out)>0.49)

        trainer.writer.add_scalar("losses/discriminator_TP", classification[:expert_obs.shape[0]].float().mean().item(), trainer.global_step)
        trainer.writer.add_scalar("losses/discriminator_TN", classification[expert_obs.shape[0]:].float().mean().item(), trainer.global_step)
        trainer.writer.add_scalar("losses/discriminator_loss", loss.item(), trainer.global_step)

    trainer.close()
    torch.save(agent,f'gail_agent_iter_{training_epochs}_mlp.pth')


if __name__ == '__main__':

    main()