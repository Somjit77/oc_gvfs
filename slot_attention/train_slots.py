"""Training loop for feature discovery with Slot Attention."""
import datetime
import time
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import random
import jax
from slot_attention.model_jax import SlotAttentionModel
import slot_attention.sa_utils as utils
import random
from tqdm import tqdm

def train(args, model_dir, seed, buffer, num_train_steps):
    random.seed(seed)
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    new_key = next(rng)
    start = time.time()
    print('Training Slot Attention')
    # Model
    model = SlotAttentionModel(args, new_key)
    # Get Buffer
    buffer = utils.get_buffer_from_exp_replay(buffer)
    batch = jnp.asarray(random.sample(buffer, args.sa_batch_size))
    batch = utils.pre_process_batch(batch, (args.sa_resolution, args.sa_resolution))
    params, optimizer_state, global_step = model.init_network(model_dir, new_key, batch)
    
    for _ in range(global_step, global_step + num_train_steps):
        batch = jnp.asarray(random.sample(buffer, args.sa_batch_size))
        batch = utils.pre_process_batch(batch, (args.sa_resolution, args.sa_resolution))

        # Learning rate warm-up.
        if global_step < args.sa_warmup_steps:
            learning_rate = args.sa_learning_rate * jnp.float32(global_step) / jnp.float32(args.sa_warmup_steps)
        else:
            learning_rate = args.sa_learning_rate
            learning_rate = learning_rate * (args.sa_decay_rate ** (
                    jnp.float32(global_step) / jnp.float32(args.sa_decay_steps)))
        # New Learning Rate
        optimizer_state.hyperparams['learning_rate'] = learning_rate

        loss_value, params, optimizer_state = model.train_slots(params, batch, next(rng), optimizer_state)
        # Update the global step. We update it before logging the loss and saving
        # the model so that the last checkpoint is saved at the last iteration.
        global_step += 1

    utils.save_model(params, model_dir, global_step)
        