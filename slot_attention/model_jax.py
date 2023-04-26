'''Code adapted from https://github.com/google-research/slot-attention-video'''
"""Slot Attention model for object discovery and set prediction."""
import logging
import haiku as hk
import jax.numpy as jnp
import jax
import time
import optax
import os
from slot_attention.sa_utils import load_model, newest

class SlotAttention(hk.Module):
    """Slot Attention module."""
#    rng_key: jax.interpreters.xla.DeviceArray

    def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
                 key, epsilon=1e-8):
        """Builds the Slot Attention module.

        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.key = key
        self.norm_inputs = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm_slots = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm_mlp = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # Parameters for Gaussian init (shared by all slots).
        mu_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        self.slots_mu = mu_init(
           [1, 1, self.slot_size],
            jnp.float32)
        sigma_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        self.slots_log_sigma = sigma_init(
            [1, 1, self.slot_size],
            jnp.float32)

        # Linear maps for the attention module.
        self.project_q = hk.Linear(self.slot_size, with_bias=False, name="q")
        self.project_k = hk.Linear(self.slot_size, with_bias=False, name="k")
        self.project_v = hk.Linear(self.slot_size, with_bias=False, name="v")

        # Slot update functions.
        self.gru = hk.GRU(self.slot_size)
        self.mlp = hk.Sequential([
            hk.Linear(self.mlp_hidden_size),
            jax.nn.relu,
            hk.Linear(self.slot_size)
        ], name="mlp")

    def __call__(self,  inputs):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + jnp.exp(self.slots_log_sigma) * jax.random.normal(
            self.key, [jnp.shape(inputs)[0], self.num_slots, self.slot_size])

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            q *= self.slot_size ** -0.5  # Normalization.
            attn_logits = jnp.einsum('ijk,ihk->ijh', k, q)
            attn = jax.nn.softmax(attn_logits, axis=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weigted mean.
            attn += self.epsilon
            attn /= jnp.sum(attn, axis=-2, keepdims=True)
            updates = jnp.einsum('ijk,ijh->ikh', attn, v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            # Slot update.
            slots, encoded_state = hk.dynamic_unroll(self.gru,updates,self.gru.initial_state(updates.shape[0]),time_major=False)
            #slots, _ = self.gru(updates,self.gru.initial_state(updates.shape[0]))
            slots += self.mlp(self.norm_mlp(slots))

        return slots


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = jnp.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = jnp.tile(slots, [1, resolution[0], resolution[1], 1])
    # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
    return grid


def spatial_flatten(x):
    return jnp.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = jnp.reshape(x, [batch_size, -1] + list(x.shape)[1:])
    c1,c2,c3,masks = jnp.split(unstacked, unstacked.shape[-1], axis=-1)
    channels = jnp.concatenate([c1,c2,c3],axis=-1)
    return channels, masks


class SlotAttentionAutoEncoder(hk.Module):
    """Slot Attention-based auto-encoder for object discovery."""

    def __init__(self, resolution, num_slots, num_iterations, key):
        """Builds the Slot Attention-based auto-encoder.

        Args:
          resolution: Tuple of integers specifying width and height of input image.
          num_slots: Number of slots in Slot Attention.
          num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.key = key
        self.encoder_cnn = hk.Sequential([
            hk.Conv2D(32, kernel_shape=5, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=5, padding="SAME"),
            jax.nn.relu,
            hk.Conv2D(64, kernel_shape=5, padding="SAME"),
            jax.nn.relu,
        ], name="encoder_cnn")

        self.decoder_initial_size = (8, 8)
        self.decoder_cnn = hk.Sequential([
            hk.Conv2DTranspose(
                64, 5, stride=(2, 2), padding="SAME"),
            jax.nn.relu,
            hk.Conv2DTranspose(
                32, 5, stride=(2, 2), padding="SAME"),
            jax.nn.relu,
            hk.Conv2DTranspose(
                32, 5, stride=(2, 2), padding="SAME"),
            jax.nn.relu,
            hk.Conv2DTranspose(
                32, 5, stride=(1, 1), padding="SAME"),
            jax.nn.relu,
            hk.Conv2DTranspose(
                4, 3, stride=(1, 1), padding="SAME")
        ], name="decoder_cnn")

        self.encoder_pos = SoftPositionEmbed(64, self.resolution)
        self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)

        self.layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.mlp = hk.Sequential([
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(64)
        ], name="feedforward")

        self.slot_attention = SlotAttention(
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=64,
            mlp_hidden_size=128,key=self.key)

    def __call__(self, image):
        # `image` has shape: [batch_size, width, height, num_channels].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = self.encoder_pos(x)  # Position embedding.
        x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
        x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)

        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=image.shape[0])
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = jax.nn.softmax(masks, axis=1)
        recon_combined = jnp.sum(recons * masks, axis=1)  # Recombine image.
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        return recon_combined, recons, masks, slots

def build_grid(resolution):
    ranges = [jnp.linspace(0., 1., num=res) for res in resolution]
    grid = jnp.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = jnp.stack(grid, axis=-1)
    grid = jnp.reshape(grid, [resolution[0], resolution[1], -1])
    grid = jnp.expand_dims(grid, axis=0)
    grid = grid.astype(jnp.float32)
    return jnp.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed():
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.dense = hk.Linear(hidden_size, with_bias=True)
        self.grid = build_grid(resolution)

    def __call__(self, inputs):
        return inputs + self.dense(self.grid)


class SlotAttentionModel:
    def __init__(self, args, key):
        '''Initialize Model'''
        self.resolution = (args.sa_resolution, args.sa_resolution)
        self.args = args
        self.optimizer_slots = optax.inject_hyperparams(optax.adam)(learning_rate=args.sa_learning_rate, eps_root=1e-8)
        self.network = self.build_forward_fn(key)
        self.train_slots = jax.jit(self.train_slots)

    def init_network(self, model_dir, key, batch, step_number=None):
        '''Build the Network'''
        if step_number is None:
            checkpoint = newest(model_dir)
        else:
            checkpoint = model_dir+f'{step_number}'
        if checkpoint is not None:
            print('Found Model')
            global_step = int(checkpoint.split('/')[-1])
            params = load_model(checkpoint)
        else:
            print('Training from Scratch')
            global_step = 0
            params = self.network.init(key, batch)
        optimizer_state = self.optimizer_slots.init(params)
        return params, optimizer_state, global_step

    def build_forward_fn(self, key) -> hk.Transformed:
        def forward_fn(batch):
            module = SlotAttentionAutoEncoder(self.resolution, self.args.sa_num_slots, self.args.sa_num_iterations, key)
            return module(batch)
        return hk.transform(forward_fn)

    def mse_loss(self, params, key, batch):
        preds = self.network.apply(params, key, batch)
        recon_combined, _, _, _ = preds
        loss = jnp.mean((recon_combined - batch)**2)
        return loss

    def train_slots(self, params, batch, key, optimizer_state):
        # Perform a single training step.
        loss, param_grads = jax.value_and_grad(self.mse_loss)(params, key, batch)
        updates, optimizer_state = self.optimizer_slots.update(param_grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return loss, params, optimizer_state
