import gym

class CoinRun:
    def __init__(self, seed, start_level, num_levels=1):
        self.env = gym.make('procgen:procgen-coinrun-v0', use_backgrounds=False, rand_seed=seed,  num_levels=num_levels,
                            start_level=start_level)
        self.num_actions = self.env.action_space.n

    def change_level(self, seed, level, num_levels=1):
        self.env = gym.make('procgen:procgen-coinrun-v0', use_backgrounds=False, rand_seed=seed,  num_levels=num_levels,
                            start_level=level)
    def reset(self):
        return {'image': self.env.reset()/255.0}
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return {'image': obs/255.0}, float(rew), done