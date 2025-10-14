# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, Autoreset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from doorsenvs import DoorsGym

from agents import DQNAgent



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="DoorsGymDQN",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-steps", type=int, default=38,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1024,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=200,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=500,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    return args


gym.register(id='DoorsGym-v0',entry_point="doorsenvs:DoorsGym",)


def make_env(seed=None, num_steps = 300):
    base_env = gym.make('DoorsGym-v0',
                max_episode_steps=num_steps,
                gridSize=[30,30],
                render_frames=False)

    env = Autoreset(base_env)
    env = RecordEpisodeStatistics(env)
    _ = env.reset(seed=seed)

    return env


# ALGO LOGIC: initialize agent here:

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    NUM_ENVS = args.num_envs

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)
    # env setup
    #envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    vecEnvs = gym.vector.SyncVectorEnv([lambda : make_env(num_steps=args.num_steps) for i in range(NUM_ENVS)])
    q_network = DQNAgent(vecEnvs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = DQNAgent(vecEnvs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        vecEnvs.observation_space,
        vecEnvs.action_space,
        device,
        n_envs=NUM_ENVS,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs,infos = vecEnvs.reset()
    dones = np.array([False]* NUM_ENVS)

    for global_step in range(args.total_timesteps//(NUM_ENVS)):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        if random.random() < epsilon:
            actions = vecEnvs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = vecEnvs.step(actions)


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos.keys():
            print(f"global_step={global_step}, episodic_return={infos['episode']['r'].mean()}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"].mean(), global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"].mean(), global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        # old dones
        not_done = np.logical_not(dones).copy()
        dones = np.logical_or(terminated,truncated)

        if not_done.any():

            rb.add(obs[not_done], 
                   real_next_obs[not_done], 
                   actions[not_done], 
                   rewards[not_done], 
                   terminated[not_done])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():

                target_max, _ = target_network(data.next_observations).max(dim=1)
                # Termination only (Truncation takes target_max)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions.int()).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    writer.close()
    torch.save(q_network,f'dqn_doorsgym_{args.total_timesteps}_mlp.pth')
    #torch.save(q_network,f'dqn_qnet_iter_{args.total_timesteps}_mlp.pth')