

from test_dm_control import EnvVectorWalker,display_video,save_video
import cv2
import copy

import torch

from ddpg_continuous_action import Actor
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

model_ddpg = 'runs/Walker2D__ddpg_continuous_action__1__1748705147/ddpg_continuous_action.cleanrl_model'
model_td3 = 'runs/Walker2d__td3_continuous_action__1__1748706498/td3_continuous_action.cleanrl_model'

def main():

    env = EnvVectorWalker(Nenvs=1,Nsteps=320)

    actor = Actor(env=env)
    model_dicts = torch.load(model_ddpg,map_location=torch.device('cuda'))
    actor.load_state_dict((model_dicts[0]))


    # Simulate episode with random actions
    duration = 8  # Seconds
    frames = []
    ticks = []
    rewards = []
    observations = []

    next_obs = env.reset()
    while env.allEnvs[0].physics.data.time < duration:

        with torch.no_grad():
            action = [actor(torch.Tensor(next_obs[0])).cpu().numpy()]
        next_obs, reward, terminations, truncations, infos = env.step(action)

        camera0 = env.allEnvs[0].physics.render(camera_id=0, height=400, width=400)
        camera1 = env.allEnvs[0].physics.render(camera_id=1, height=400, width=400)
        frames.append(np.hstack((camera0, camera1)))
        rewards.append(reward[0])
        observations.append(copy.deepcopy(next_obs[0]))
        ticks.append(env.allEnvs[0].physics.data.time)
        if truncations[0]:
            breakpoint()
    display_video(frames, framerate=1./env.allEnvs[0].control_timestep())
    save_video(frames, framerate=1./env.allEnvs[0].control_timestep())



if __name__ == '__main__':

    main()
