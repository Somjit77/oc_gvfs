import numpy as np
import jax.numpy as jnp
from collections import namedtuple
from pygame_gridworld import PyGame
import gym
import gym_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper

class CoinRun:
    def __init__(self, seed, start_level, num_levels=1):
        self.env = gym.make('procgen:procgen-coinrun-v0', use_backgrounds=False, rand_seed=seed,  num_levels=num_levels,
                            start_level=start_level, distribution_mode="easy")
        self.num_actions = self.env.action_space.n

    def change_level(self, seed, level, num_levels=1):
        self.env = gym.make('procgen:procgen-coinrun-v0', use_backgrounds=False, rand_seed=seed,  num_levels=num_levels,
                            start_level=level)
    def reset(self):
        return {'image': self.env.reset()/255.0}
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return {'image': obs/255.0}, float(rew), done

class StarPilot:
    def __init__(self, seed, start_level, num_levels=1):
        self.env = gym.make('procgen:procgen-starpilot-v0', use_backgrounds=False, rand_seed=seed, num_levels=num_levels,
                            start_level=start_level, distribution_mode="easy")
        self.num_actions = self.env.action_space.n

    def change_goal(self, seed, level, num_levels=1):
        self.env = gym.make('procgen:procgen-starpilot-v0', use_backgrounds=False, rand_seed=seed, num_levels=num_levels,
                            start_level=level)

    def reset(self):
        return {'image': self.env.reset() / 255.0}

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return {'image': obs / 255.0}, float(rew), done

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = StarPilot(seed=0, start_level=30)
    state = env.reset()
    done = False
    count = 0
    while not done:
        action = env.env.action_space.sample()
        state, reward, done = env.step(action)
        if count == 50:
            plt.imshow(state['image'])
            plt.savefig('test0.png')
            exit(0)
        import time
        time.sleep(1)
        count += 1
