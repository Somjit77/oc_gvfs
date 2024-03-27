"""Experiment loop."""

import haiku as hk
import jax
from jax.config import config
from tensorboardX import SummaryWriter
from dm_env import TimeStep, StepType
from tqdm import tqdm
import gym
import numpy as np
import json, argparse
from pathlib import Path
from agent_procgen import ReplayBuffer, DQN
from env_procgen import CoinRun, StarPilot
from slot_attention.train_slots import train as train_slots
import utils
import os

parser = argparse.ArgumentParser()
parser.add_argument("--game", help='environment', default='CoinRun')  # Environment Name
parser.add_argument("--algorithm", help='algorithm', default='sa_esp')  # dqn, gvf, hc_gvf, gvf_esp, hc_gvf_esp
parser.add_argument("--version", help='version', default='0.0')
parser.add_argument("--runs", type=int, default=1, help="Number of Runs.")
parser.add_argument("--level", type=int, default=0, help="Level in procgen")
parser.add_argument("--train_episodes", type=int, default=5001, help="Number of train episodes.")
parser.add_argument("--transfer_episodes", type=int, default=5001,
                    help="Number of episodes after which task transitions.")
parser.add_argument("--batch_size", type=int, default=32, help="Size of the training batch")
parser.add_argument("--target_period", type=float, default=100, help="How often to update the target net.")
parser.add_argument("--replay_capacity", type=int, default=10000, help="Capacity of the replay buffer.")
parser.add_argument("--hidden_arch", type=int, default=[64, 32], help="Number of network hidden units.")
parser.add_argument("--epsilon_begin", type=float, default=1., help="Initial epsilon-greedy exploration.")
parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon-greedy exploration.")
parser.add_argument("--epsilon_steps", type=int, default=1.0,
                    help="portion of total episodes over which to anneal eps.")
parser.add_argument("--discount_factor", type=float, default=0.99, help="Q-learning discount factor.")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Optimizer learning rate.")
parser.add_argument("--eval_episodes", type=int, default=100, help="Number of evaluation episodes.")
parser.add_argument("--evaluate_every", type=int, default=50, help="Number of episodes between evaluations.")
# GVFs
parser.add_argument("--num_gvfs", type=int, default=5, help="Number of GVFs.")
parser.add_argument("--use_action_values", action='store_true', help="Use action values for GVFs")
parser.add_argument("--hand_crafted_cumulants", type=bool, default=False, help="Use Hand Crafted Cumulants")
parser.add_argument("--use_off_policy", action='store_true', help="Off-policy Q-learning for training GVFs")
# Discovery
parser.add_argument("--discovery", type=bool, default=False, help="Discovery of Cumulants")
parser.add_argument("--unroll_steps", type=int, default=10, help="Unroll steps for meta-gradient")
parser.add_argument("--use_concatanation", type=bool, default=True, help="Use Concatanation for ESP")
# Slot Attention Args
parser.add_argument("--use_slot_attention", type=bool, default=False, help="Use Slot Attention?")
parser.add_argument("--sa_batch_size", type=int, default=16, help='Batch size for the model.')
parser.add_argument("--sa_resolution", type=int, default=64, help='Resolution of the Image')
parser.add_argument("--sa_num_slots", type=int, default=5, help="Number of slots in Slot Attention.")
parser.add_argument("--sa_num_iterations", type=int, default=3, help="Number of attention iterations.")
parser.add_argument("--sa_learning_rate", type=float, default=0.0004, help="Learning rate for Slot Attention")
parser.add_argument("--sa_num_train_steps", type=int, default=100000, help="Number of training steps.")
parser.add_argument("--sa_warmup_steps", type=int, default=10000, help="Number of warmup steps for the learning rate.")
parser.add_argument("--sa_decay_rate", type=float, default=0.5, help="Rate for the learning rate decay.")
parser.add_argument("--sa_decay_steps", type=int, default=100000, help="Number of steps for the learning rate decay.")

args = parser.parse_args()
if args.algorithm == 'dqn':
    args.num_gvfs = 0
if 'sa' in args.algorithm:
    args.use_slot_attention = True
    args.discovery = True
    args.sa_num_slots = args.num_gvfs
if 'dis' in args.algorithm:
    args.discovery = True
if 'hc' in args.algorithm:
    args.hand_crafted_cumulants = True
episodes_before_transfer = args.transfer_episodes

def run_loop(agent, environment, accumulator, seed, args):
    """A simple run loop for examples of reinforcement learning with rlax."""
    # Log and Model Directory
    if args.use_slot_attention:
        model_dir = f'Results/logs-{args.version}/sa_model_{seed}/'
        os.makedirs(model_dir, exist_ok=True)
    else:
        model_dir = None
    # Logging files
    logdir = f'Results/logs-{args.version}/'
    Path(logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir=logdir + f"{args.algorithm}_run_{seed}/")
    with open(logdir + f'{args.algorithm}_params.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Init agent.
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    new_key = next(rng)
    params = agent.initial_params(new_key)
    learner_states = []
    for gvf in range(args.num_gvfs):
        learner_states.append(agent.initial_gvf_learner_state(gvf=gvf, params=params))
    dqn_learner_state = agent.initial_dqn_learner_state(params=params)
    qn_params = agent.initial_qn_params(model_dir, new_key)
    qn_learner_state = agent.initial_qn_learner_state(params=qn_params)

    print(f"Training agent for {args.train_episodes} episodes")
    train_step = 0
    frames = 0
    slot_train_count = 0
    for episode in tqdm(range(1, args.train_episodes)):
        # Reset Environment
        state = environment.reset()
        timestep = TimeStep(step_type=StepType.FIRST, reward=0.0, discount=1.0, observation=state)
        accumulator.push(timestep, None)
        actor_state = agent.initial_actor_state()
        returns_train = 0.
        while not timestep.last():
            # Acting.
            new_key = next(rng)
            actor_output, actor_state, epsilon = agent.actor_step(
                params, timestep, actor_state, episode, new_key, evaluation=False)
            # Agent-environment interaction.
            action = int(actor_output.actions)
            state, reward, done = environment.step(action)
            frames += 1
            returns_train += reward

            if not done:
                timestep = TimeStep(step_type=StepType.MID, reward=reward, discount=1.0, observation=state)
            else:
                timestep = TimeStep(step_type=StepType.LAST, reward=reward, discount=0.0, observation=state)

            # Accumulate experience.
            accumulator.push(timestep, action)

            # Slot Attention Training
            if frames % 1000 == 999 and slot_train_count < args.sa_num_train_steps:
                if args.use_slot_attention:
                    slot_train_count += 10000
                    train_slots(args, model_dir, seed, accumulator, 10000)
                    agent.load_slot_attention_model(model_dir, args, key, None, slot_train_count)

            # Learning
            min_samples = args.batch_size + args.unroll_steps if args.discovery else args.batch_size
            if accumulator.is_ready(min_samples):
                key = next(rng)
                if args.discovery:
                    # Train Question and Main Network
                    cumulants = None
                    if train_step % args.unroll_steps == 0:
                        qn_data = accumulator.get_multiple_samples(args.batch_size, args.discount_factor,
                                                                   args.unroll_steps)
                        qn_loss, qn_params, qn_learner_state, params, learner_states, dqn_learner_state, losses = agent.question_train(
                            params,
                            qn_params, qn_data, cumulants, dqn_learner_state, learner_states, qn_learner_state, key,
                            train=True)
                    # Train Main Network
                    else:
                        data = accumulator.sample(args.batch_size, args.discount_factor)
                        qn_loss, _, _, params, learner_states, dqn_learner_state, losses = agent.question_train(params,
                                                                                                                qn_params,
                                                                                                                data,
                                                                                                                cumulants,
                                                                                                                dqn_learner_state,
                                                                                                                learner_states,
                                                                                                                qn_learner_state,
                                                                                                                key,
                                                                                                                train=False)
                else:
                    data = accumulator.sample(args.batch_size, args.discount_factor)
                    if args.hand_crafted_cumulants:
                        cumulants = utils.get_cumulants(data, args.batch_size, args.num_gvfs)
                    else:
                        cumulants = None
                    qn_loss, _, _, params, learner_states, dqn_learner_state, losses = agent.question_train(params,
                                                                                                            qn_params,
                                                                                                            data,
                                                                                                            cumulants,
                                                                                                            dqn_learner_state,
                                                                                                            learner_states,
                                                                                                            qn_learner_state,
                                                                                                            key,
                                                                                                            train=False)
                train_step += 1
                # Write Losses
                gvf_losses = np.asarray(losses[-1])
                for gvf in range(args.num_gvfs):
                    writer.add_scalar(f'GVF-{gvf} Loss', gvf_losses[gvf], train_step)
                writer.add_scalar('Training_Loss', np.asarray(losses[0]), train_step)
        # Episode Ends
        writer.add_scalar('Training_returns', returns_train, episode)
        # Plot epsilon schedule
        writer.add_scalar('Epislon', epsilon, episode)
        # Evaluation.
        if not episode % args.evaluate_every:
            returns = 0.
            for _ in range(args.eval_episodes):
                state = environment.reset()
                timestep = TimeStep(step_type=StepType.FIRST, reward=0.0, discount=1.0, observation=state)
                actor_state = agent.initial_actor_state()

                while not timestep.last():
                    actor_output, actor_state, _ = agent.actor_step(
                        params, timestep, actor_state, episode, next(rng), evaluation=True)
                    state, reward, done = environment.step(int(actor_output.actions))

                    if not done:
                        timestep = TimeStep(step_type=StepType.MID, reward=reward, discount=1.0, observation=state)
                    else:
                        timestep = TimeStep(step_type=StepType.LAST, reward=reward, discount=0.0, observation=state)
                    returns += reward

            avg_returns = returns / args.eval_episodes
            print(f"Episode {episode:4d}: Average returns: {avg_returns:.4f}")
            writer.add_scalar('Average_Return', avg_returns, episode)
    writer.flush()
    save_dir = f'Results/logs-{args.version}/models/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpointer = utils.Checkpointer(save_dir + f'{args.algorithm}_run_{seed}_{episode}')
    checkpointer.save(params)


def train(args):
    for run in range(1): #parallel runs
        seed = args.runs + 1000
        level = args.level
        if args.game == 'CollectObjects':
            env = CollectObjects(seed, breadth=7, length=7)
        elif 'MiniGrid' in args.game:
            env = MiniGrid(args.game, seed)
        elif args.game == 'CoinRun':
            env = CoinRun(seed, level, 50)
        elif args.game == 'StarPilot':
            env = StarPilot(seed, level, 50)
        else:
            raise Exception('Environment Not Defined')
        # Setting Numpy seed for random reward
        np.random.seed(seed)
        init_state = env.reset()['image']
        action_space = env.num_actions
        epsilon_cfg = dict(
            init_value=args.epsilon_begin,
            end_value=args.epsilon_end,
            transition_steps=int(args.epsilon_steps * episodes_before_transfer),
            power=1.)
        agent = DQN(
            observation_spec=init_state,
            action_spec=action_space,
            epsilon_cfg=epsilon_cfg,
            args=args
        )
        accumulator = ReplayBuffer(args.replay_capacity, seed)
        run_loop(
            agent=agent,
            environment=env,
            accumulator=accumulator,
            seed=seed,
            args=args
        )


if __name__ == "__main__":
    train(args)
