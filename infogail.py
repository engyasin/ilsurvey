
import numpy as np
import cv2

from doorsenvs import DoorZ


from agents import PPOAgentwithZ,Discriminator,Qnet
import torch
import torch.optim as optim

from ppo_class_z import ppo_trainer_z




def main():

    traj_len = 32
    training_epochs = 80
    policy_epochs = 10000
    discriminator_epochs = 16

    Qnet_epochs = 64
    demo_size = 480*3

    Z_classes = 3


    lr = 3e-5
    lr_d = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DoorZ(max_steps=traj_len)
    env.single_observation_space = (env.single_observation_space[0],)

    agent = PPOAgentwithZ(envs=[env]).to(device=device).float()
    discriminator = Discriminator(envs=[env]).to(device=device).float()

    qnet = Qnet(envs=[env],classes=Z_classes).to(device=device).float()
    optimizer_q = optim.Adam(qnet.parameters(), lr=lr_d, eps=3e-4)
    #optimizer = optim.Adam(agent.actor.parameters(), lr=lr, eps=1e-5)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, eps=1e-5)
    discriminator.agent = agent

    expert_actions = torch.as_tensor(np.load(f'expert_actions_15_{traj_len}_Z.npy').flatten(),device=device).long()[:demo_size]
    onehot_expert_acts = torch.functional.F.one_hot(expert_actions,num_classes=5).float()
    expert_obs = torch.as_tensor(np.load(f'expert_obses_15_{traj_len}_225_Z.npy').reshape(-1,np.prod(env.gridsize)),device=device).float()[:demo_size,:]


    agent.actor.train()
    discriminator.train()

    env.discriminator = discriminator
    env.qnet = qnet

    # TODO show the real reward
    trainer = ppo_trainer_z(agent=agent, lr=lr, total_timesteps=policy_epochs, buffer_size=expert_obs.shape[0], env_id= "InfoGAIL")
    # different envs
    for env_i,_ in enumerate(trainer.envs):
        trainer.envs[env_i].discriminator = discriminator
        trainer.envs[env_i].qnet = qnet
        trainer.envs[env_i].single_observation_space = (env.single_observation_space[0],)
    
    trainer.reset()

    for epoch in range(training_epochs):

        # policy training

        generator_obs, generator_acts, generated_logs,generator_z = trainer.train()

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


        for epoch_q in range(Qnet_epochs):

            out = qnet(training_batch_obs[expert_obs.shape[0]:],training_batch_acts[expert_obs.shape[0]:])
            loss = torch.functional.F.binary_cross_entropy(out,generator_z)
            optimizer_q.zero_grad()
            loss.backward()
            optimizer_q.step()

        trainer.writer.add_scalar("losses/qnet_loss", loss.item(), trainer.global_step)


    trainer.close()
    torch.save(agent,f'infogail_agent_iter_{training_epochs}_mlp.pth')
    torch.save(discriminator,f'infogail_disc_iter_{training_epochs}_mlp.pth')
    torch.save(qnet,f'infogail_qnet_iter_{training_epochs}_mlp.pth')


if __name__ == '__main__':

    main()