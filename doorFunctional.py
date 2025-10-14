
import PIL.Image
import gymnasium as gym
import cv2

from functools import partial

import PIL
import jax
from jax import jit,random
import jax.numpy as np
from jax import lax,vmap, pmap

import numpy
# create a functional copy of DoorsEnv
# accelerate it with jax

# change the following

# 1. np to jnp
# 2. for step, reset, reward, and render use: @partial(jit,static_argnums=(0,))
# 3. pass env_state = (state,key) to step
# 4. pass key to reset (which initialize state and return a new key)
# 5. use the form x = x.at[idx].set(y) for jax arrays
# 6. change control statements (if, while, switch) to lax (or not?)

# NOTE currently removing goal location -> state




class DoorsEnvJax(gym.Env):

    def __init__(self,gridSize=[15,15],nDoors=3):
        super().__init__()

        EnvConfig = {}
        self.gridSize = gridSize
        self.nDoors = nDoors

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.MultiDiscrete([4 for _ in range(self.gridSize[0]*self.gridSize[1])])

        self.actions_vocal = np.array([[0,0],[0,1],[1,0],[0,-1],[-1,0]]).astype(int)



    @partial(jit,static_argnums=(0,))
    @partial(vmap,in_axes=(None,0))
    def reset(self,key):
        agent_location = np.array([self.gridSize[0]-1,random.randint(key, shape=(1,), minval=0, maxval=self.gridSize[1])[0]])
        goal_location = np.array([0,random.randint(key, shape=(1,), minval=0, maxval=self.gridSize[1])[0]])

        new_key = random.split(key)[0,:]
        state = self._make_grid(agent_location,goal_location)
        info = self._get_info(agent_location,goal_location)
        #if self.render_frames:
        #    self.render(state)
        info.update({"new_state":np.zeros_like(state[None,...]),
                     "episode":{'r':0,'l':0},
                     "num_steps":35})

        return (state,new_key),info

    @partial(jit,static_argnums=(0,))
    #@partial(vmap,in_axes=(None,0,0))
    def _make_grid(self,agent_location,goal_location):

        grid = np.zeros(tuple(self.gridSize),dtype=np.int32)
        grid = grid.at[self.gridSize[0]//2].set(2)
        grid = grid.at[self.gridSize[0]//2,::(self.gridSize[0]//self.nDoors)].set(0)
        # put agent and goal
        grid = grid.at[tuple(agent_location)].set(1)
        grid = grid.at[tuple(goal_location)].set(3)
        #grid = grid.at[agent_location[0],agent_location[1]].set(1)
        #grid = grid.at[goal_location[0],goal_location[1]].set(3)
        return grid

    @partial(jit,static_argnums=(0,))
    #@partial(vmap,in_axes=(None,0,0))
    def _get_info(self,agent_location,goal_location):

        return {'distance': np.linalg.norm(goal_location-agent_location),
                'agent_location': agent_location.copy(),
                'goal_location': goal_location}

    @partial(jit,static_argnums=(0,))
    @partial(vmap,in_axes=(None,0,0,0))
    def step(self, action, env_state, info):

        key = env_state[1]
        state = env_state[0]
        agent_location = info['agent_location']
        goal_location = info['goal_location']
        episodic_reward = info['episode']['r']
        timestep = info['episode']['l']
        max_steps = info["num_steps"]


        movement = self.actions_vocal[action]
        #new_location = agent_location+movement
        new_location = np.clip(agent_location+movement,0,np.array(self.gridSize)-1)

        terminated = False
        truncated = np.array(max_steps<=timestep,dtype=np.bool_)
        past_position = agent_location.copy()

        # check if wall (2)

        cell_state = state.at[*tuple(new_location)].get()
        #lax.cond(cell_state in [0,3],)
        #lax.cond()

        possible_moves = np.logical_or(cell_state == 0, cell_state == 3) 
        state = np.where(possible_moves,
                state.at[tuple(agent_location)].set(0).at[tuple(new_location)].set(1),
                state
                 )
        #if possible_moves.any():
        #state[possible_moves] = state[possible_moves].at[agent_location[possible_moves]].set(0)
        #state[possible_moves] = state[possible_moves].at[new_location[possible_moves]].set(1)

        #if cell_state in [0,3]:
        #    state = state.at[agent_location].set(0)
        #    state = state.at[new_location].set(1)

        agent_location = new_location.copy()

        terminated = (cell_state == 3) 

        #terminated = (cell_state == 3) 
        reward = self._get_reward(past_position,agent_location,goal_location)
        info.update(self._get_info(agent_location,goal_location))

        # automatic reset
        new_state = np.where(np.logical_or(terminated,truncated),
                 self.reset(key[None,:])[0][0][0,...], # to remove vector dimension
                (state).copy())

        info.update({"new_state":new_state,
                     "episode":{'r':episodic_reward+reward,'l':timestep+1},
                     "agent_location":np.hstack(np.where(new_state==1,size=1)),
                     "goal_location":np.hstack(np.where(new_state==3,size=1))})
        #             "new_info": new_info})
        #if self.render_frames:
        #    self.render(state)
        new_key = random.split(key)[0,:]

        return (new_state,new_key), reward, terminated, truncated, info

    @partial(jit,static_argnums=(0,))
    def _get_reward(self,past_location,agent_location,goal_location):

        old_distance = np.linalg.norm(goal_location-past_location)
        new_distance = np.linalg.norm(goal_location-agent_location)
        return old_distance - new_distance

    @partial(jit,static_argnums=(0,))
    @partial(vmap,in_axes=(None,0))
    def render(self,state, scale=10):
        grid = state
        img = np.dstack([grid==3,grid==2,grid==1])*1.0#.astype(np.uint16)
        # make it big enough
        img = img.repeat(scale,axis=0).repeat(scale,axis=1)
        # NOTE: Error here
        return img
        #PIL.Image.fromarray(numpy.asarray(img)).show()
        #cv2.imshow('Doors',numpy.asarray(img))

    def close(self):

        cv2.destroyAllWindows()



from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

from gymnasium.utils.env_checker import check_env

import time 

#gym.register(id='DoorsGym-v0',entry_point="doorsenvs:DoorsGym",)

def make_env():

    base_env = DoorsEnvJax(nDoors=3,
                gridSize=[30,30],
                )
    #base_env = TimeLimit(base_env, max_episode_steps=25)
    # DoorsEnvJax
    #env = RecordEpisodeStatistics(base_env)

    return base_env

def run_env(env):

    key = random.PRNGKey(0)
    NUM_DEVICES = 1 # pmap
    NUM_ENVS = 4 # vmap
    keys = random.split(key,NUM_ENVS)#.reshape(NUM_DEVICES, NUM_ENVS//NUM_DEVICES,-1)

    #EnvConfigs = [init_env(nDoors=3,gridSize=[30,30],render_frames=False) for _ in range(NUM_ENVS)]

    #envs = [make_env() for _ in range(NUM_ENVS)]
    env_state, info = env.reset(keys)
    state = env_state[0]
    print(state.device)
    start_time = time.time()
    for i in range(250):

        action = np.array([env.action_space.sample() for _ in range(NUM_ENVS)])
        env_state, reward, terminated, truncated, info = env.step(action, env_state,
                                                              info)
        #print(action)
        #print(info['agent_location'])
        #print(info['agent_location']-info['goal_location'])
        #imgs = env.render(env_state[0])
        #cv2.imshow('out',numpy.asarray(imgs[NUM_ENVS-1,...]))
        cv2.waitKey(10)
        if np.array([truncated]).any():
            print(f'truncated at step {i}')
            #print(info)
            break

    print("Time taken to step in the environment:", time.time() - start_time)
    env.close()

@partial(jit,static_argnums=(1,))
def run_jax_env(key,NUM_ENVS):

    env = DoorsEnvJax(nDoors=3,
                gridSize=[30,30],
                )
    #env = Autoreset(base_env)
    #env = RecordEpisodeStatistics(env)
    env_state, info = env.reset(key)# to emulate patch

    def for_loop_body(i,init_val):
        env_state, reward, terminated, truncated, info = init_val
        new_key = random.split(env_state[1][0,:])[0]
        action = random.randint(new_key,(NUM_ENVS,),minval=0,maxval=5)
        #print(action)
        #action = np.array([env.action_space.sample() for _ in range(NUM_ENVS)])
        env_state, reward, terminated, truncated, info = env.step(action, env_state,
                                                              info)
        #cv2.waitKey(10)
        #if np.where(truncated,lambda x: True, lambda x: False, ):#truncated.any():
        #    print(f'truncated at step {i}')
            #print(info)
        #    break
        return env_state, reward, terminated, truncated, info

    init_val = (env_state, np.zeros((NUM_ENVS,)) , np.zeros((NUM_ENVS,),dtype=np.bool_),
                 np.zeros((NUM_ENVS,),dtype=np.bool_),  info)

    env_state, reward, terminated, truncated, info = lax.fori_loop(0,2500, for_loop_body, init_val)



def main():


    #print(gym.envs.registry.keys())#
    #print(gym.pprint_registry())

    #env = make_env()

    #check_env(env.unwrapped)

    #num_envs = 4
    #vecEnvs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

    start_time = time.time()

    #run_env(env)
    key = random.PRNGKey(0)
    NUM_DEVICES = 1 # pmap
    NUM_ENVS = 2400 # vmap
    keys = random.split(key,NUM_ENVS)#.reshape(NUM_DEVICES, NUM_ENVS//NUM_DEVICES,-1)

    jax.block_until_ready(run_jax_env(keys,NUM_ENVS))
    print("Time taken to run the environment:", time.time() - start_time)
    #SyncVectorEnv : 0.00698 s
    #AsyncVectorEnv : 0.01801 s


if __name__ == '__main__':
    main()