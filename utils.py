import numpy as np
import tensorflow as tf
import jax, pickle

#HC Cumulants
def get_cumulants(data, batch_size, num_gvfs, breadth=9, length=9):
    cumulants = np.zeros([batch_size, num_gvfs])
    assert num_gvfs < 6
    for batch in range(batch_size):
        prev_state = data[-2][batch]
        next_state = data[-1][batch]
        # Red Dot
        if next_state[-1] == 1 and prev_state[-1] == 0:
            cumulants[batch, 0] = 1
        else:
            cumulants[batch, 0] = 0
        # Bottom Left Quadrant
        if next_state[0] <= breadth//2 and next_state[1] <= length//2:
            cumulants[batch, 1] = 1
        else: 
            cumulants[batch, 1] = 0
        # Top Left Quadrant
        if next_state[0] <= breadth//2 and next_state[1] > breadth//2:
            cumulants[batch, 2] = 1
        else: 
            cumulants[batch, 2] = 0
        # Bottom Right Quadrant
        if next_state[0] > breadth//2 and next_state[1] <= length//2:
            cumulants[batch, 3] = 1
        else:
            cumulants[batch, 3] = 0
        # Top Right Quadrant
        if next_state[0] > breadth//2 and next_state[1] > length//2:
            cumulants[batch, 4] = 1
        else: 
            cumulants[batch, 4] = 0
    return cumulants

@jax.jit       
def td_error(q_tm1, a_tm1, cumulant, discount_t, q_t_val, q_t_select):
    """Based on given inputs, calculates a TD error"""
    target_tm1 = cumulant + discount_t * jax.lax.stop_gradient(q_t_val[q_t_select.argmax()])
    return target_tm1 - q_tm1[a_tm1]

@jax.jit       
def td_error_state(v_tm1, cumulant, discount_t, v_t_val):
    """Based on given inputs, calculates a TD error"""
    target_tm1 = cumulant + discount_t * jax.lax.stop_gradient(v_t_val)
    return target_tm1 - v_tm1

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