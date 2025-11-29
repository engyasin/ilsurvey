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

import jax
import mlflow
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from doorFunctional import DoorsEnvJax
from functools import partial

from doorsenvs import DoorsGym

from agents import DQNAgent
import cv2


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
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-steps", type=int, default=38,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-envs", type=int, default=256*4,
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
    parser.add_argument("--learning-starts", type=int, default=100,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--train-iteration", type=int, default=8,
        help="the iteration of training")
    args = parser.parse_args()
    # fmt: on
    return args


gym.register(id='DoorsGym-v0',entry_point="doorsenvs:DoorsGym",)


def make_env(seed=None, num_steps = 300):
    base_env = gym.make('DoorsGym-v0',
                max_episode_steps=num_steps,
                gridSize=[15,15],
                render_frames=False)

    env = Autoreset(base_env)
    env = RecordEpisodeStatistics(env)
    _ = env.reset(seed=seed)

    return env


# ALGO LOGIC: initialize agent here:

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def objective(trial,argsParams,device):

    # define hyper parameters

    argsParams.update({"num_steps":trial.suggest_int("num_steps", 10, 17, step=1)})
    argsParams.update({"learning_rate":trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)})
    argsParams.update({"buffer_size":trial.suggest_int("buffer_size", 128, 1024, step=32)})
    argsParams.update({"batch_size":trial.suggest_int("batch_size", 16, 128, step=16)})
    argsParams.update({"train_frequency":trial.suggest_int("train_frequency", 4, 24, step=1, log=True)})
    argsParams.update({"optimizer_name": trial.suggest_categorical("optimizer_name", ["Adam", "SGD"])})
    argsParams.update({"train_iteration":trial.suggest_int("train_iteration", 4, 16, step=1, log=False)})

    # init networks, optimizer, env and buffer

    #run_env(env)
    key = jax.random.PRNGKey(argsParams['seed'])
    key, q_key = jax.random.split(key, 2)
    keys = jax.random.split(key,argsParams['num_envs'])#.reshape(NUM_DEVICES, NUM_ENVS//NUM_DEVICES,-1)
    env = DoorsEnvJax(nDoors=3,
                gridSize=[15,15],
                )
    
    env_state, infos = env.reset(keys)# to emulate patch
    obs,keys = env_state
    obs = obs.reshape(-1,np.array(env.observation_space.shape).prod())
    obs = jax.device_get(obs)
    infos["num_steps"] = infos["num_steps"].at[:].set(argsParams["num_steps"])


    q_network = DQNAgent(env).to(device)
    
    # Configure optimizer
    if argsParams["optimizer_name"] == "Adam":
        optimizer = optim.Adam(q_network.parameters(),  lr=argsParams["learning_rate"])
    else:
        optimizer = optim.SGD(q_network.parameters(),  lr=argsParams["learning_rate"])

    target_network = DQNAgent(env).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        argsParams["buffer_size"],
        env.observation_space,
        env.action_space,
        device,
        n_envs=argsParams["num_envs"],
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game

    dones = np.array([False]* argsParams["num_envs"])
    rolling_rewards = [0]



    with mlflow.start_run(nested=True) as run:

        mlflow.log_params(argsParams)

        for global_step in range(argsParams["total_timesteps"]//(argsParams['num_envs'])):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(argsParams["start_e"], argsParams["end_e"], argsParams["exploration_fraction"] * (argsParams["total_timesteps"]//argsParams['num_envs']), global_step)

            if random.random() < epsilon:
                key = jax.random.split(key)[0]
                actions = jax.random.randint(key,(argsParams['num_envs'],),minval=0,maxval=5)
                actions = jax.device_get(actions)
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            env_state, rewards, terminated, truncated, infos = env.step(actions, env_state ,infos)
            next_obs,new_key = env_state
            next_obs = next_obs.reshape(-1,np.array(env.observation_space.shape).prod())
            next_obs = jax.device_get(next_obs)
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            finished = np.logical_or(terminated,truncated)

            if finished.any():

                print(f"global_step={global_step}, episodic_return={infos['episode']['r'][finished].mean()}")

                mlflow.log_metrics({"charts/episodic_return": infos["episode"]["r"][finished].mean(),
                                    "charts/episodic_length": infos["episode"]["l"][finished].mean(),
                                    "charts/epsilon": f"{epsilon:2f}"},
                                    step=global_step)

                rolling_rewards.append(infos["episode"]["r"][finished].mean()/infos["episode"]["l"][finished].mean())

                infos["episode"]["r"] = infos["episode"]["r"].at[finished].set(0)
                infos["episode"]["l"] = infos["episode"]["l"].at[finished].set(0)
            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`

            real_next_obs = next_obs.copy()

            #cv2.imshow('a',((next_obs[0,...]*80)).astype(np.uint8).reshape(15,15))
            #cv2.waitKey(5)
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
            if global_step > argsParams["learning_starts"] and global_step % argsParams["train_frequency"] == 0:
                for i in range(argsParams["train_iteration"]):
                    data = rb.sample(argsParams["batch_size"])
                    with torch.no_grad():

                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        # Termination only (Truncation takes target_max)
                        td_target = data.rewards.flatten() + argsParams["gamma"] * target_max * (1 - data.dones.flatten())
                    old_val = q_network(data.observations).gather(1, data.actions.int()).squeeze()
                    loss = F.mse_loss(td_target, old_val)


                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_step % 100 == 0:

                    mlflow.log_metric("losses/td_loss", loss, step=global_step)
                    mlflow.log_metric("losses/q_values", old_val.mean().item(), step=global_step)
                    mlflow.log_metric("losses/SPS", int(global_step / (time.time() - start_time)), step=global_step)

                    print("SPS:", int(global_step / (time.time() - start_time)))

                # update the target network
                if global_step % argsParams["target_network_frequency"] == 0:
                    target_network.load_state_dict(q_network.state_dict())

            # for hyperband
            trial.report(np.mean(rolling_rewards[-500:]), step=global_step)
            mlflow.log_metric("rolling_reward", np.mean(rolling_rewards[-500:]), step=global_step)

            if trial.should_prune():
                raise optuna.TrialPruned()


        model_info = mlflow.pytorch.log_model(q_network, name=f'dqn_doorsgym_{argsParams["total_timesteps"]}_mlp')

        return np.mean(rolling_rewards[-500:])
    


def main():

    args = parse_args()
    experiment_name = f"{args.env_id}__{args.exp_name}"
    run_name = f"{args.env_id}_{args.seed}__{int(time.time())}"

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(f"runs/{experiment_name}")


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    max_epochs = args.total_timesteps//args.num_envs
    # Execute hyperparameter search
    with mlflow.start_run(run_name=run_name) as run:

        study = optuna.create_study(sampler=TPESampler(seed=args.seed, multivariate=False),
                                    pruner=HyperbandPruner(min_resource=240, max_resource=max_epochs, reduction_factor=3), #resource represents epochs
                                    direction="maximize")

        objective_func = partial(
            objective, argsParams=vars(args).copy(), device=device
        )

        study.optimize(objective_func, n_trials=40)

        # Log best parameters and score
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_reward", study.best_value)

    # load best model
    ranked_models = mlflow.search_logged_models(#experiment_ids=[f"runs/{experiment_name}"],
                                                filter_string=f"source_run_id='{run.info.run_id}'",
                                                order_by=[{"field_name": "metrics.rolling_reward", "ascending": False}],
                                                output_format="list",
                                                )

    # Get the best performing model
    best_model = ranked_models#[0]

    model_uri = f"models:/{best_model.model_id}" # or from dashboard
    model = mlflow.pytorch.load_model(model_uri)



if __name__ == "__main__":

    main()


    