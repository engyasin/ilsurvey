

import copy
import cv2
import time
import numpy as np
import gymnasium as gym

from dm_control import suite

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation




class EnvVectorWalker():

    def __init__(self,Nenvs=20,Nsteps=160,domain='walker',task='walk'):

        self.allEnvs = []
        self.Nsteps = Nsteps
        self.domain = domain
        self.task = task
        self.Nenvs = Nenvs
        self.num_envs = Nenvs

        for i in range(self.Nenvs):

            self.allEnvs.append(suite.load(domain, task, 
                         task_kwargs={'random': np.random.RandomState(42+i)}))
            
            time_step = self.allEnvs[i].reset()
            
        self.timeStep = self.allEnvs[0].control_timestep()

        self.obsKeys = time_step.observation.keys()
        obs = self.flat_obs(time_step)
        self.single_observation_space = gym.spaces.Box(low=-(np.inf),
                                                        high=(np.inf),
                                                        shape=(len(obs),),
                                                        dtype=np.float32)

        single_action_space = self.allEnvs[0].action_spec()
        self.single_action_space = gym.spaces.Box(low=single_action_space.minimum, 
                                                  high=single_action_space.maximum, 
                                                  shape=single_action_space.shape, 
                                                  dtype=np.float32)

        self.reward = 0
        self.done = False


    def step(self,actions):

        observations = []
        rewards = []
        dones = []

        infos = []

        self.episodic_length += 1
        for j,action in zip(range(self.Nenvs),actions):

            time_step = self.allEnvs[j].step(action)
            observations.append(self.flat_obs(time_step))

            #observations.append([time_step.observation[k] for k in self.obsKeys])
            rewards.append(time_step.reward)
            self.episodic_rewards[j] += time_step.reward

            if time_step.last():

                dones.append(True)
                infos.append({'episode':{'r':self.episodic_rewards[j],'l':self.episodic_length},
                              'final_observation':observations[-1]})


            else:
                dones.append(False)
                infos.append({})
            
        if dones[0]:
            observations = self.reset()

        return observations,rewards, dones, dones, infos

    def flat_obs(self,timestep):

        full_obs = []
        for k in self.obsKeys:
            full_obs.extend(timestep.observation[k].flatten().tolist())
        return full_obs

    def reset(self,seed=0):

        observations = []

        self.episodic_rewards = [0 for _ in self.allEnvs]
        self.episodic_length = 0


        for env in self.allEnvs:
            time_step = env.reset()
            observations.append(self.flat_obs(time_step))

        return observations

    def close(self):
        pass


def display_video(video, framerate):

    cv2.namedWindow('out')
    for frame in video:

        cv2.imshow('out',frame)
        
        k = cv2.waitKey(int((1/framerate) * 1000))
        if k== ord('q'):
            break
    
    cv2.destroyAllWindows()



def test_env(domain='walker',task='walk'):



    random_state = np.random.RandomState(42)

    env = suite.load(domain, task, task_kwargs={'random': random_state})

    # Simulate episode with random actions
    duration = 4  # Seconds
    frames = []
    ticks = []
    rewards = []
    observations = []

    spec = env.action_spec()
    time_step = env.reset()
    while env.physics.data.time < duration:

        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        time_step = env.step(action)

        camera0 = env.physics.render(camera_id=0, height=400, width=400)
        camera1 = env.physics.render(camera_id=1, height=400, width=400)
        frames.append(np.hstack((camera0, camera1)))
        rewards.append(time_step.reward)
        observations.append(copy.deepcopy(time_step.observation))
        ticks.append(env.physics.data.time)

    #display_video(frames, framerate=1./env.control_timestep())

    # Show video and plot reward and observations (each item for a variable)
    num_sensors = len(time_step.observation)

    _, ax = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
    ax[0].plot(ticks, rewards)
    ax[0].set_ylabel('reward')
    ax[-1].set_xlabel('time')

    for i, key in enumerate(time_step.observation):
        # extract sensor by sensor (key) for all time steps
        data = np.asarray([observations[j][key] for j in range(len(observations))])
        ax[i+1].plot(ticks, data, label=key)
        ax[i+1].set_ylabel(key)

    plt.show()






def save_video(frames, framerate=30):

    height, width, _ = frames[0].shape
    dpi = 70

    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)

    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')

    ax.set_position([0, 0, 1, 1])

    im = ax.imshow(frames[0])

    def update(frame):

        im.set_data(frame)
        return [im]

    interval = 1000/framerate

    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=interval, blit=True, repeat=False)

    writer = animation.PillowWriter(fps=50,
                                     metadata=dict(artist='yasin'),
                                     bitrate=1800)

    anim.save('Walker.gif', writer=writer)






if __name__ == '__main__':

    test_env()


