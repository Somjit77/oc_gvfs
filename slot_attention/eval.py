import argparse
import jax.numpy as jnp
import numpy as np
import haiku as hk
import random
import jax
jax.config.update('jax_platform_name', 'cpu')
from slot_attention.model_jax import SlotAttentionModel
import slot_attention.sa_utils as utils
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
#Slot Attention Args
parser.add_argument("--sa_batch_size", type=int, default=1, help='Batch size for the model.')
parser.add_argument("--sa_resolution", type=int, default=32, help='Resolution of the Image')
parser.add_argument("--sa_num_slots", type=int, default=5, help="Number of slots in Slot Attention.")
parser.add_argument("--sa_num_iterations", type=int, default=3, help="Number of attention iterations.")
parser.add_argument("--sa_learning_rate", type=float, default=0.0004, help="Learning rate for Slot Attention")
parser.add_argument("--sa_num_train_steps", type=int, default=100000, help="Number of training steps.")
parser.add_argument("--sa_warmup_steps", type=int, default=10000, help="Number of warmup steps for the learning rate.")
parser.add_argument("--sa_decay_rate", type=float, default=0.5, help="Rate for the learning rate decay.")
parser.add_argument("--sa_decay_steps", type=int, default=100000, help="Number of steps for the learning rate decay.")
args = parser.parse_args()

goals = {1: [(9, 9), (9, 1)], 2: [(7, 7), (7, 3)], 3: [(3, 6), (9, 9)]}
goals = {1: [(9, 9), (9, 1)]}

def main(args, vers):
    for ver in vers:
        runs = 5
        print(ver)
        for run in range(runs):
            for goal_no in range(1):
                seed = 100
                ckpt_path = f"Results/logs-{ver}/sa_model_{run+1000}/"
                random.seed(seed)
                rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
                if goal_no == 0:
                    buffer = utils.get_buffer_from_env(seed=seed, goal=None, replay_size=100)
                    load_model_number = 400000
                elif goal_no == 1:
                    buffer = utils.get_buffer_from_env(seed=seed, goal=1, replay_size=500)
                    load_model_number = None
                new_key = next(rng)
                batch = jnp.asarray(random.sample(buffer, args.sa_batch_size))
                batch = utils.pre_process_batch(batch, (args.sa_resolution, args.sa_resolution))
                # load_model_number = None #10000
                model = SlotAttentionModel(args, new_key)
                try:
                    params, _, global_step = model.init_network(ckpt_path, new_key, batch, load_model_number)
                except FileNotFoundError: #when the directory doesn't exist
                    print('Failed to Load Model')
                    continue
                if global_step == 0: #when directory exists but the model doesn't
                    print('Failed to Load Model')
                    continue
                else:
                    print(f'Loaded Trained Model at {global_step} training steps')

                for id in range(5):
                    # Get new batch.
                    batch = jnp.asarray(random.sample(buffer, args.sa_batch_size))
                    batch = utils.pre_process_batch(batch, (args.sa_resolution, args.sa_resolution))
                    # Predict.
                    image, recon_combined, recons, masks, slots = utils.get_prediction(model.network, params, next(rng), batch)
                    image, recon_combined, recons, masks = image[0], recon_combined[0], recons[0], masks[0]
                    # Visualize.
                    num_slots = len(masks)
                    fig, ax = plt.subplots(1, args.sa_num_slots + 2, figsize=(15, 2))
                    ax[0].imshow(image)
                    ax[0].set_title('Image')
                    ax[1].imshow(np.clip(recon_combined, 0, 1))
                    ax[1].set_title('Recon.')
                    for i in range(num_slots):
                        ax[i + 2].imshow(np.clip(recons[i] * masks[i] + (1 - masks[i]), 0, 1))
                        ax[i + 2].set_title('Slot %s' % str(i + 1))
                    for i in range(len(ax)):
                        ax[i].grid(False)
                        ax[i].axis('off')
                    save_dir = f'plots/slot_attention/{ver}-{run}/'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(save_dir+f'goal_{goal_no}_sample{id+1}.png')
                    plt.close(fig)

if __name__=='__main__':
    vers = np.arange(11,15)
    final_vers = []
    for ver in vers:
      final_vers.append('9.'+str(ver))
    final_vers = ['19.5']
    main(args, final_vers)
