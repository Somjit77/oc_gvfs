import collections
import random
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import utils
import slot_attention.sa_utils as sa_utils
from slot_attention.model_jax import SlotAttentionModel

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Params = collections.namedtuple("Params", "online target")
ActorState = collections.namedtuple("ActorState", "count")
ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
LearnerState = collections.namedtuple("LearnerState", "count opt_state")

class Model(hk.Module):
    """Network Architecture"""
    def __init__(self, output_size, num_gvfs, hidden_units, algorithm, use_action_values, use_concatanation, name='model'):
        super().__init__(name=name)
        self.num_gvfs = num_gvfs
        self.algorithm = algorithm
        self.output_size = output_size
        self.use_action_values = use_action_values
        self.hidden_units = hidden_units
        self.use_concatanation = use_concatanation
        '''Representation'''
        self._cnn_layers = []
        self._cnn_layers.append(hk.Conv2D(output_channels=16, kernel_shape=(3,3), name='rep_1'))
        self._cnn_layers.append(hk.MaxPool(window_shape=2, strides=2, padding='SAME', name='rep_2'))
        self._cnn_layers.append(hk.Conv2D(output_channels=32, kernel_shape=(3,3), name='rep_3'))
        self._cnn_layers.append(hk.MaxPool(window_shape=2, strides=2, padding='SAME', name='rep_4'))
        self._cnn_layers.append(hk.Conv2D(output_channels=64, kernel_shape=(3,3), name='rep_5'))
        self._rep_layers = []
        self._rep_layers.append(hk.Flatten())
        self._rep_layers.append(hk.nets.MLP(output_sizes=hidden_units, activate_final=True, name='rep_dense'))
        '''GVF Outputs'''
        self._gvf_output_layers = []
        for gvf in range(self.num_gvfs):
            if self.use_action_values:
                self._gvf_output_layers.append(hk.nets.MLP(output_sizes=[output_size], name=f'gvf_{gvf}'))
            else:
                self._gvf_output_layers.append(hk.nets.MLP(output_sizes=[1], name=f'gvf_{gvf}')) 
        if 'esp' in self.algorithm:
            '''ESP'''
            self._layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='esp_norm')
            self._projection_layer = hk.nets.MLP(output_sizes=[32], name='esp_proj')
            self._output_layer = hk.nets.MLP(output_sizes=[output_size], name='esp')
        else:
            '''Original DQN'''
            self._output_layer = hk.nets.MLP(output_sizes=[output_size], name='dqn')

    def __call__(self, x):
        outputs = []
        rep = x
        # Representation Model
        for layer_no in range(len(self._cnn_layers)):
            rep = self._cnn_layers[layer_no](rep)
            rep = jax.nn.relu(rep)
        for layer_no in range(len(self._rep_layers)):
            rep = self._rep_layers[layer_no](rep)
        # GVF Models
        for gvf in range(self.num_gvfs):
            outputs.append(self._gvf_output_layers[gvf](rep))
        outputs.append(rep) #last item contains the representation
        # Output Model
        if 'esp' in self.algorithm: 
            '''Concatenate all GVF Outputs'''
            if self.use_action_values:
                output_gvf = jnp.reshape(jnp.asarray(outputs[:-1]), newshape=[-1, self.output_size*self.num_gvfs])
            else:
                output_gvf = jnp.reshape(jnp.asarray(outputs[:-1]), newshape=[-1, self.num_gvfs])
            if self.use_concatanation:
                # Projected Concatenation
                gvf_transform = self._projection_layer(output_gvf)
                output_gvf = jnp.concatenate([gvf_transform, rep], axis=-1)
                output_gvf = self._layer_norm(output_gvf)
            dqn_output = self._output_layer(output_gvf)
        else:
            dqn_output = self._output_layer(rep) # outputs from the main RL agent

        return dqn_output, outputs

class QuestionNetwork(hk.Module):
    """Network Architecture"""
    def __init__(self, output_size, name='QuestionNetwork'):
        super().__init__(name=name)
        self._qn_layer = hk.nets.MLP(output_sizes=[64, output_size], activation=jax.nn.tanh, activate_final=True, name='qn')
    
    def __call__(self, x):
        return self._qn_layer(x)
    
class ReplayBuffer(object):
    """Experience replay buffer."""
    def __init__(self, capacity, seed):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)
        random.seed(seed)

    def push(self, env_output, action):
        self._prev = self._latest
        self._action = action
        self._latest = env_output

        if action is not None:
            self.buffer.append(
                    (self._prev.observation['image'], self._action, self._latest.reward, self._latest.discount, 
                     self._latest.observation['image'], self._prev.observation['position'], self._latest.observation['position']))

    def sample(self, batch_size, discount_factor):
        obs_tm1, a_tm1, r_t, discount_t, obs_t, s_tm1, s_t = zip(
                *random.sample(self.buffer, batch_size))
        return (np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
                        np.asarray(discount_t) * discount_factor, np.stack(obs_t), np.stack(s_tm1), np.stack(s_t))

    def get_multiple_samples(self, batch_size, discount_factor, unroll_steps):
        indices = random.sample(range(0, len(self.buffer)-unroll_steps), batch_size)
        data = []
        for k in range(unroll_steps):
            obs_tm1, a_tm1, r_t, discount_t, obs_t, s_tm1, s_t = [],[],[],[],[],[],[]
            for idx in indices:
                _obs_tm1, _a_tm1, _r_t, _discount_t, _obs_t, _s_tm1, _s_t = self.buffer[idx+k]
                obs_tm1.append(_obs_tm1)
                a_tm1.append(_a_tm1)
                r_t.append(_r_t)
                discount_t.append(_discount_t)
                obs_t.append(_obs_t)
                s_tm1.append(_s_tm1)
                s_t.append(_s_t)
            data_batch = (np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
                        np.asarray(discount_t) * discount_factor, np.stack(obs_t), np.stack(s_tm1), np.stack(s_t))
            assert len(data_batch[0]) == batch_size
        data.append(data_batch) 
        return data

    def reshape(self, batch_size, discount_factor, unroll_steps):
        data = []
        for k in range(unroll_steps):
            obs_tm1, a_tm1, r_t, discount_t, obs_t, s_tm1, s_t = [],[],[],[],[],[],[]
            for idx in range(0, len(self.buffer), unroll_steps):
                _obs_tm1, _a_tm1, _r_t, _discount_t, _obs_t, _s_tm1, _s_t = self.buffer[idx+k]
                obs_tm1.append(_obs_tm1)
                a_tm1.append(_a_tm1)
                r_t.append(_r_t)
                discount_t.append(_discount_t)
                obs_t.append(_obs_t)
                s_tm1.append(_s_tm1)
                s_t.append(_s_t)
            data_batch = (np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
                        np.asarray(discount_t) * discount_factor, np.stack(obs_t), np.stack(s_tm1), np.stack(s_t))
            assert len(data_batch[0]) == batch_size
        data.append(data_batch) 
        return data

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)


def build_network(num_actions: int, num_gvfs: int, hidden_units: int, algorithm: str, 
                  use_action_values: bool, use_concatanation: bool) -> hk.Transformed:
    """Build the Main Network"""
    def forward_pass(x):
        module = Model(num_actions, num_gvfs, hidden_units, algorithm, use_action_values, use_concatanation)
        return module(x)
    forward = hk.transform(forward_pass)
    return forward

def build_question_network(num_gvfs: int):
    """Build the Question Network"""
    def forward_pass(x):
        module = QuestionNetwork(num_gvfs)
        return module(x)
    forward= hk.transform(forward_pass)
    return forward


class DQN:
    """Main Agent"""
    def __init__(self, observation_spec, action_spec, epsilon_cfg, args):
        # Environment Params
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self.resolution = (args.sa_resolution, args.sa_resolution)
        # Type of Algorithm
        self.algorithm = args.algorithm
        self.num_gvfs = args.num_gvfs
        self.discovery = args.discovery
        self.hc = args.hand_crafted_cumulants
        self.use_sa = args.use_slot_attention
        self.off = args.use_off_policy
        self.use_action_values = args.use_action_values
        # Algorithm Params
        self.target_period = args.target_period
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.rep_size = args.hidden_arch[-1]
        self.args = args
        # Neural net and optimisers for each gvf.
        self._network = build_network(action_spec, self.num_gvfs, args.hidden_arch, self.algorithm, self.use_action_values, args.use_concatanation)
        if 'esp' in self.algorithm:
            self._dqn_optimizer = optax.adam(args.learning_rate, eps_root=1e-8)
        else:
            self._dqn_optimizer = optax.adam(args.learning_rate, eps_root=1e-8)
        self._optimizers = []
        for _ in range(self.num_gvfs):
            self._optimizers.append(optax.adam(args.learning_rate, eps_root=1e-8))
        if self.use_sa:
            self._question_network = build_question_network(1)
            self._qn_optimizer = optax.adam(args.learning_rate, eps_root=1e-8)
        else:
            self._question_network = build_question_network(self.num_gvfs)
            self._qn_optimizer = optax.adam(args.learning_rate, eps_root=1e-8)
        # Episilon Schedule
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
        # Jitting for speed.
        self.actor_step = jax.jit(self.actor_step)
        self.question_train = jax.jit(self.question_train, static_argnums=[8])


    def initial_params(self, key):
        sample_input = self._observation_spec
        sample_input = jnp.expand_dims(sample_input, 0)
        online_params = self._network.init(rng=key, x=sample_input)
        return Params(online_params, online_params)

    def initial_qn_params(self, model_dir, key):
        sample_input = self._observation_spec
        sample_input = jnp.expand_dims(sample_input, 0)
        if self.use_sa:
            batch = sa_utils.pre_process_batch(sample_input, self.resolution)
            self.load_slot_attention_model(model_dir, self.args, key, batch)
            features = jnp.asarray(self.get_slots(key, sample_input))
        else:
            features = jnp.zeros(shape=(self.batch_size, self.rep_size))
        qn_params = self._question_network.init(rng=key, x=features)
        return qn_params

    def initial_actor_state(self):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return ActorState(actor_count)

    def initial_dqn_learner_state(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        if 'esp' not in self.algorithm:
            trainable_params, non_trainable_params = hk.data_structures.partition(
                lambda m, n, p: 'dqn' in m or 'rep' in m, params.online)
        else:
            trainable_params, non_trainable_params = hk.data_structures.partition(
                lambda m, n, p: 'esp' in m or 'rep' in m, params.online)
        opt_state = self._dqn_optimizer.init(trainable_params)
        return LearnerState(learner_count, opt_state)
    
    def transfer_dqn(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        if 'esp' not in self.algorithm:
            trainable_params, non_trainable_params = hk.data_structures.partition(
                lambda m, n, p: 'dqn' in m, params.online)
        else:
            trainable_params, non_trainable_params = hk.data_structures.partition(
                lambda m, n, p: 'esp' in m, params.online)
        opt_state = self._dqn_optimizer.init(trainable_params)
        return LearnerState(learner_count, opt_state)

    def transfer_gvf(self, gvf, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        trainable_params, non_trainable_params = hk.data_structures.partition(
            lambda m, n, p: f'gvf_{gvf}' in m, params.online)
        opt_state = self._optimizers[gvf].init(trainable_params)
        return LearnerState(learner_count, opt_state)

    def initial_gvf_learner_state(self, gvf, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        trainable_params, non_trainable_params = hk.data_structures.partition(
            lambda m, n, p: f'gvf_{gvf}' in m or 'rep' in m, params.online)
        opt_state = self._optimizers[gvf].init(trainable_params)
        return LearnerState(learner_count, opt_state)

    def initial_qn_learner_state(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        trainable_params, non_trainable_params = hk.data_structures.partition(lambda m, n, p: 'qn' in m, params)
        opt_state = self._qn_optimizer.init(trainable_params)
        return LearnerState(learner_count, opt_state)

    def actor_step(self, params, env_output, actor_state, episode, key, evaluation):
        obs = jnp.expand_dims(env_output.observation['image'], 0)  # add dummy batch
        q, gvf_outputs = self._network.apply(params.online, key, obs) 
        q = q[0] # remove dummy batch
        epsilon = self._epsilon_by_frame(episode)
        train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
        eval_a = rlax.greedy().sample(key, q)
        a = jax.lax.select(evaluation, eval_a, train_a)
        return ActorOutput(actions=a, q_values=q), ActorState(actor_state.count + 1), epsilon

    def qn_loss(self, params, qn_params, data, cumulants, dqn_learner_state, learner_states, key):
        total_loss = 0
        gvf_losses = []
        for step in range(len(data)):
            '''Get Cumulants'''
            if self.use_sa:
                representation = self.get_slots(key, data[step][0])
                cumulants = jnp.squeeze(self._question_network.apply(qn_params, key, jnp.asarray(representation)))
            elif self.discovery and (not self.use_sa):
                _, gvf_outputs = self._network.apply(params.online, key, jnp.asarray(data[step][0]))
                representation = gvf_outputs[-1]
                cumulants = jnp.squeeze(self._question_network.apply(qn_params, key, jnp.asarray(representation)))
            elif 'hc' in self.algorithm:
                cumulants = cumulants
            else:
                _, gvf_outputs = self._network.apply(params.online, key, jnp.asarray(data[step][0]))
                representation = gvf_outputs[-1]
                cumulants = jnp.squeeze(self._question_network.apply(qn_params, key, jnp.asarray(representation)))
            '''Target Refresh'''
            target_params = optax.periodic_update(params.online, params.target, dqn_learner_state.count, self.target_period)
            '''GVF Training'''
            for gvf in range(self.num_gvfs):
                trainable_params, non_trainable_params = hk.data_structures.partition(
                    lambda m, n, p: f'gvf_{gvf}' in m or 'rep' in m, params.online)
                if self.use_action_values:
                    loss, dloss_dtheta = jax.value_and_grad(self._gvf_loss_q, 1)(key, trainable_params, non_trainable_params, target_params, gvf, cumulants[:, gvf], *data[step])
                else:
                    loss, dloss_dtheta = jax.value_and_grad(self._gvf_loss_v, 1)(key, trainable_params, non_trainable_params, target_params, gvf, cumulants[:, gvf], *data[step])
                updates, opt_state = self._optimizers[gvf].update(dloss_dtheta, learner_states[gvf].opt_state)
                online_trainable_params = optax.apply_updates(trainable_params, updates)
                learner_states[gvf] = LearnerState(learner_states[gvf].count + 1, opt_state)
                online_params = hk.data_structures.merge(online_trainable_params, non_trainable_params)
                gvf_losses.append(loss)
                params = Params(online_params, target_params)
            '''Main RL Training'''
            if 'esp' in self.algorithm:
                trainable_params, non_trainable_params = hk.data_structures.partition(
                    lambda m, n, p: 'esp' in m or 'rep' in m, params.online)
            else:
                trainable_params, non_trainable_params = hk.data_structures.partition(
                    lambda m, n, p: 'dqn' in m or 'rep' in m, params.online)
            loss, dloss_dtheta = jax.value_and_grad(self._dqn_loss, 1)(key, trainable_params, non_trainable_params, target_params, *data[step])
            updates, opt_state = self._dqn_optimizer.update(dloss_dtheta, dqn_learner_state.opt_state)
            online_trainable_params = optax.apply_updates(trainable_params, updates)
            dqn_learner_state = LearnerState(dqn_learner_state.count + 1, opt_state)
            online_params = hk.data_structures.merge(online_trainable_params, non_trainable_params)
            params = Params(online_params, target_params)
            total_loss += self._main_loss(key, online_params, target_params, *data[step])
        return total_loss, (params, learner_states, dqn_learner_state, (loss, gvf_losses))
    
    def question_train(self, params, qn_params, data, cumulants, dqn_learner_state, learner_states, qn_learner_state, key, train=False):
        if train: #train question network
            (loss, aux), dloss_deta = jax.value_and_grad(self.qn_loss, 1, has_aux=True)(params, qn_params, data, cumulants, dqn_learner_state, learner_states, key)
            updates, opt_state = self._qn_optimizer.update(dloss_deta, qn_learner_state.opt_state)
            qn_params = optax.apply_updates(qn_params, updates)
            qn_learner_state = LearnerState(qn_learner_state.count+1, opt_state)
        else: #train main network only
            data = [data] # no concatenation here
            loss, aux = self.qn_loss(params, qn_params, data, cumulants, dqn_learner_state, learner_states, key)
        return (loss, qn_params, qn_learner_state, *aux) 

    def apply_updates(self, weights, grads):
        '''Manual Gradient Step'''
        for key in weights.keys():
            for wb in weights[key].keys():
                weights[key][wb] -= self.learning_rate * grads[key][wb]
        return weights
 
    def _main_loss(self, key, online_params, target_params, obs_tm1, a_tm1, r_t, discount_t, obs_t, s_tm1, s_t):
        q_tm1, _ = self._network.apply(online_params, key, obs_tm1)
        q_t_val, _ = self._network.apply(target_params, key, obs_t)
        q_t_select, _ = self._network.apply(online_params, key, obs_t)
        batched_loss = jax.vmap(utils.td_error) #Double Q Learning
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))

    def _dqn_loss(self, key, trainable_params, non_trainable_params, target_params, obs_tm1, a_tm1, r_t, discount_t, obs_t, s_tm1, s_t):
        online_params = hk.data_structures.merge(trainable_params, non_trainable_params)
        q_tm1, _ = self._network.apply(online_params, key, obs_tm1)
        q_t_val, _ = self._network.apply(target_params, key, obs_t)
        q_t_select, _ = self._network.apply(online_params, key, obs_t)
        batched_loss = jax.vmap(utils.td_error) #Double Q Learning
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))
    
    def _gvf_loss_q(self, key, trainable_params, non_trainable_params, target_params, gvf, cumulant, obs_tm1, a_tm1, r_t, discount_t,
                        obs_t, s_tm1, s_t):
        online_params = hk.data_structures.merge(trainable_params, non_trainable_params)
        _, q_tm1 = self._network.apply(online_params, key, obs_tm1)
        q_tm1 = q_tm1[gvf] # take each GVF output
        _, q_t_val = self._network.apply(target_params, key, obs_t)
        q_t_val = q_t_val[gvf] # take each GVF output
        q_t_dqn, q_t_select = self._network.apply(online_params, key, obs_t)
        if self.off:
            # Off-Policy
            q_t_select = q_t_select[gvf] # take each GVF output
        else:
            # On-Policy
            q_t_select = q_t_dqn
        batched_loss = jax.vmap(utils.td_error) #Double Q Learning
        td_error = batched_loss(q_tm1, a_tm1, cumulant, discount_t, q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))

    def _gvf_loss_v(self, key, trainable_params, non_trainable_params, target_params, gvf, cumulant, obs_tm1, a_tm1, r_t, discount_t,
                        obs_t, s_tm1, s_t):
        online_params = hk.data_structures.merge(trainable_params, non_trainable_params)
        _, v_tm1 = self._network.apply(online_params, key, obs_tm1)
        v_tm1 = v_tm1[gvf] # take each GVF output
        _, v_t_val = self._network.apply(target_params, key, obs_t)
        v_t_val = v_t_val[gvf] # take each GVF output
        batched_loss = jax.vmap(utils.td_error_state) # TD Error with Action Values
        td_error = batched_loss(v_tm1, cumulant, discount_t, v_t_val)
        return jnp.mean(rlax.l2_loss(td_error))
    
    def get_slots(self, key, data):
        states = sa_utils.pre_process_batch(data, resolution=self.resolution)
        image, recon_combined, recons, masks, slots = sa_utils.get_prediction(self.sa_model, self.sa_params, key, states)
        return slots

    def load_slot_attention_model(self, model_dir, args, key, input, step_number=None):
        sa_class = SlotAttentionModel(args, key)
        self.sa_params, _, _ = sa_class.init_network(model_dir, key, input, step_number)
        self.sa_model = sa_class.network