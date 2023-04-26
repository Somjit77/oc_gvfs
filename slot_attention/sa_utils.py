import jax
from env import CollectObjects, MiniGrid
import numpy as np
import pickle, os
import haiku as hk

class Checkpointer:
  def __init__(self, path):
    self.path = path

  def save(self, params):
    params = jax.device_get(params)
    with open(self.path, 'wb') as fp:
      pickle.dump(params, fp)

  def load(self):
    with open(self.path, 'rb') as fp:
      params = pickle.load(fp)
    return jax.device_put(params)

# Resize image to 84x84 and convert to (-1,1)
def pre_process_batch(batch, resolution=(48,48)):
    batch = (batch - 0.5) * 2.0  # Rescale to [-1, 1].
    orig_shape = batch.shape
    batch = jax.image.resize(batch, [orig_shape[0], resolution[0], resolution[1], orig_shape[3]], method="cubic") #resize image to same size for all grid sizes
    batch = jax.numpy.clip(batch, -1., 1.) # Rescale to [-1. 1]
    return batch

def renormalize(x):
    """Renormalize from [-1, 1] to [0, 1]."""
    return x / 2. + 0.5

#Get latest file
def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    if len(paths) == 0:
      return None #No model saved
    else:
      return max(paths, key=os.path.getctime)

# Get predictions from the slot attention model
def get_prediction(network, params, key, batch):
    recon_combined, recons, masks, slots = network.apply(params, key, batch)
    image = renormalize(batch)
    recon_combined = renormalize(recon_combined)
    recons = renormalize(recons)
    masks = masks
    return image, recon_combined, recons, masks, slots

def get_buffer_from_env(seed, goal=None, replay_size=100):
    env = CollectObjects(seed, 7, 7)
    # env = MiniGrid('MiniGrid-Dynamic-Obstacles-Random-6x6-v0', 0)
    '''Returns a buffer of states by executing random actions'''
    if goal is not None:
      env.add_obstacle()  
    buffer = []
    state = env.reset(True)
    for _ in range(replay_size):
        action = np.random.randint(env.num_actions)
        new_state, reward, done = env.step(action)
        image = state['image']
        buffer.append(image)
        if done:
            new_state = env.reset()
        state = new_state.copy()
    return buffer

def get_buffer_from_exp_replay(accumulator):
  buffer = []
  for id in range(len(accumulator.buffer)):
    image = accumulator.buffer[id][0]
    buffer.append(image)
  return buffer

# Load Slot Attention Model
def load_model(model_dir):
  checkpointer = Checkpointer(model_dir)
  params = checkpointer.load()
  return params

def save_model(params, model_dir, global_step):
  checkpointer = Checkpointer(model_dir+str(global_step))
  #Checkpoint save writes parameters to disk.
  checkpointer.save(params)