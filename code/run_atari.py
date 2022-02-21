#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# parts of this code are copied from DQN_ZOO https://github.com/deepmind/dqn_zoo
# under the Apache License, Version 2.0 (the "License");


"""A *-{C|Q}R-DQN agent training on Atari.
Paper: A Cramér Distance perspective on Quantile Regression based Distributional Reinforcement Learning
https://arxiv.org/abs/2110.00535
""" 

import collections
import itertools
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import dm_env
import haiku as hk
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np
import optax

from dqn_zoo import atari_data
from dqn_zoo import gym_atari
from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib


from typing import Any, Callable, Mapping, Text

import rlax

import chex
Array = chex.Array
Numeric = chex.Numeric

FLOAT_TYPE =  jnp.float32



def qr_atari_network(num_actions: int, num_quantiles: int) -> networks.NetworkFn:
  """QR-DQN network, expects `uint8` input."""


  def net_fn(inputs):
    """Function representing QR-DQN Q-network."""
    network = hk.Sequential([
        networks.dqn_torso(),
        networks.dqn_value_head(num_quantiles * num_actions),
    ])
    network_output = network(inputs)
    q_dist = jnp.reshape(network_output, (-1, num_quantiles, num_actions))
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return networks.QRNetworkOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def nc_atari_network(num_actions: int, num_quantiles: int, 
                        n_layers: int, n_nodes: int ) -> networks.NetworkFn:

  def net_fn(inputs):
    """Function representing NC-QR-DQN Q-network."""

    torso_output = networks.dqn_torso()(inputs)

    N = num_quantiles

    layers_Q0_AMP = []
    layers_QPROP = []

    layers = [layers_Q0_AMP, layers_QPROP]

    for _ in range(n_layers):
      for l in layers:
        l += [networks.linear(n_nodes), jax.nn.relu]
        
    layers_Q0_AMP += [networks.linear(2*num_actions)]
    layers_QPROP += [networks.linear(N *num_actions)]

    network_output_Q0_AMP = hk.Sequential(layers_Q0_AMP)(torso_output)
    network_output_QPROP = hk.Sequential(layers_QPROP)(torso_output)
    
    # slice and reshape to have the action as the last dimension
    Rq = jnp.reshape(network_output_QPROP, (-1, N, num_actions))
    
    Q0 =  jnp.reshape(network_output_Q0_AMP[:,0:num_actions], (-1, 1, num_actions))
    AMP =  jnp.reshape(network_output_Q0_AMP[:,num_actions:], (-1, 1, num_actions))

    AMP = jax.nn.relu(AMP)
        
    Qprop = jax.nn.softmax(Rq, axis=1)

    Q = jnp.cumsum(Qprop,axis=1)


    Q *= AMP
   
    q_dist = Q +  Q0 
      
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return networks.QRNetworkOutputs(q_dist=q_dist, q_values=q_values)


  return net_fn




def cramer_dist(
    dist_src: Array,
    dist_target: Array,
    #temp_param: float #unused
) -> Numeric:
    
  chex.assert_rank([dist_src, dist_target], 1)
  chex.assert_type([dist_src, dist_target], float)
       
  #num_quantiles
  n = dist_src.shape[0]

  tau_inc = jnp.ones(n, dtype=FLOAT_TYPE)/n

  y_diff = jnp.concatenate([-tau_inc,tau_inc],axis=0)
  Qs = jnp.concatenate([dist_src,dist_target],axis=0)

  (sorted_Qs, y_diff) = jax.lax.sort((Qs,y_diff))    


  x_diff = sorted_Qs[1:] - sorted_Qs[:-1]
  y_diff = jnp.cumsum(y_diff)[:-1]

  integs = y_diff * x_diff * y_diff
  return jnp.sum(integs)


from rlax._src import clipping
Array = chex.Array
Numeric = chex.Numeric

def _quantile_regression_loss(
    dist_src: Array,
    #tau_src: Array,
    dist_target: Array,
    huber_param: float = 0.
) -> Numeric:
  """Compute (Huber) QR loss between two discrete quantile-valued distributions.
  See "Distributional Reinforcement Learning with Quantile Regression" by
  Dabney et al. (https://arxiv.org/abs/1710.10044).
  Args:
    dist_src: source probability distribution.
    tau_src: source distribution probability thresholds.
    dist_target: target probability distribution.
    huber_param: Huber loss parameter, defaults to 0 (no Huber loss).
  Returns:
    Quantile regression loss.
  """
  
  N = dist_src.shape[0]
  tau_src = (jnp.arange(0, N) + 0.5) / float(N)
  
  chex.assert_rank([dist_src, tau_src, dist_target], 1)
  chex.assert_type([dist_src, tau_src, dist_target], float)

  # Calculate quantile error.
  delta = dist_target[None, :] - dist_src[:, None]
  delta_neg = (delta < 0.).astype(jnp.float32)
  delta_neg = jax.lax.stop_gradient(delta_neg)
  weight = jnp.abs(tau_src[:, None] - delta_neg)

  # Calculate Huber loss.
  if huber_param > 0.:
    loss = clipping.huber_loss(delta, huber_param)
  else:
    loss = jnp.abs(delta)
  loss *= weight

  # Average over target-samples dimension, sum over src-samples dimension.
  return jnp.sum(jnp.mean(loss, axis=-1))



def quantile_q_learning(
    dist_q_tm1: Array,
    #tau_q_tm1: Array,
    a_tm1: Numeric,
    r_t: Numeric,
    discount_t: Numeric,
    dist_q_t_selector: Array,
    dist_q_t: Array,
    q_values: Array,
    cramer: bool,
    huber_param: Numeric
) -> Numeric:
  """Implements Q-learning for quantile-valued Q distributions.
  See "Distributional Reinforcement Learning with Quantile Regression" by
  Dabney et al. (https://arxiv.org/abs/1710.10044).
  Args:
    dist_q_tm1: Q distribution at time t-1.
    tau_q_tm1: Q distribution probability thresholds.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    dist_q_t_selector: Q distribution at time t for selecting greedy action in
      target policy. This is separate from dist_q_t as in Double Q-Learning, but
      can be computed with the target network and a separate set of samples.
    dist_q_t: target Q distribution at time t.
    huber_param: Huber loss parameter, defaults to 0 (no Huber loss).
  Returns:
    Quantile regression Q learning loss.
  """
  chex.assert_rank([
      dist_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t
  ], [2, 0, 0, 0, 2, 2])
  chex.assert_type([
      dist_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t
  ], [float, int, float, float, float, float])

  # Only update the taken actions.
  dist_qa_tm1 = dist_q_tm1[:, a_tm1]

    
  # Select target action according to greedy policy w.r.t. dist_q_t_selector.
  #q_t_selector = jnp.mean(dist_q_t_selector, axis=0)
  ## q_values == q_t_selector.val : TRUE
  q_t_selector = q_values 
      
  
  a_t = jnp.argmax(q_t_selector)

  dist_qa_t = dist_q_t[:, a_t]

  # Compute target, do not backpropagate into it.
  dist_target = r_t + discount_t * dist_qa_t
  dist_target = jax.lax.stop_gradient(dist_target)

  if cramer: 
    return cramer_dist(dist_qa_tm1, dist_target) #, temp_param)
  else:
    return _quantile_regression_loss(dist_qa_tm1, dist_target, huber_param) #1.)



# Batch variant of quantile_q_learning with fixed tau input across batch.
_batch_quantile_q_learning = jax.vmap(
    quantile_q_learning, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))



class QrDqn:
  """Quantile Regression DQN agent."""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      #quantiles: jnp.ndarray,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      cramer: bool,
      huber_param: float,
      rng_key: parts.PRNGKey,
	  num_quantiles: int,
	  scale_grad: bool
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input[None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.


    # Define jitted loss, update, and policy functions here instead of as
    # class methods, to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.



    def loss_fn(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      # Compute Q value distributions.
      _, online_key, target_key = jax.random.split(rng_key, 3)
      dist_q_tm1 = network.apply(online_params, online_key,
                                 transitions.s_tm1).q_dist
      
      q_target_t = network.apply(target_params, target_key,
                                      transitions.s_t)
      
      dist_q_target_t  = q_target_t.q_dist
      
      #this could be used instead of recomputing the mean
      q_values_target_t  = q_target_t.q_values


      losses = _batch_quantile_q_learning(
          dist_q_tm1,
          #quantiles,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          dist_q_target_t,  # No double Q-learning here.
          dist_q_target_t,
          q_values_target_t,
          cramer,
          huber_param
      )
      assert losses.shape == (self._batch_size,)
      loss = jnp.mean(losses)
      return loss


    def update(rng_key, opt_state, online_params, target_params, transitions):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)

      d_loss_d_params = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key)
      if scale_grad:
          #scale gradient to make it equivalent to Cramer/QR depending on the loss used
          scaling_factor = 2./num_quantiles if not cramer else num_quantiles/2.
          d_loss_d_params = jax.tree_multimap(lambda x:x*scaling_factor,d_loss_d_params)
          

      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params


    self._update = jax.jit(update)
 


    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = rlax.epsilon_greedy().sample(policy_key, q_t, exploration_epsilon)
      return rng_key, a_t

    self._select_action = jax.jit(select_action)


  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      action = self._action
    else:
      action = self._action = self._act(timestep)

      for transition in self._transition_accumulator.step(timestep, action):
        self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation
    self._rng_key, a_t = self._select_action(self._rng_key, self._online_params,
                                             s_t, self.exploration_epsilon)
    return parts.Action(jax.device_get(a_t))

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions = self._replay.sample(self._batch_size)
    
    self._rng_key, self._opt_state, self._online_params = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions, 
    )

  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def exploration_epsilon(self) -> float:
    """Returns epsilon value currently used by (eps-greedy) behavior policy."""
    return self._exploration_epsilon(self._frame_t)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])
########################################################################

# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'pong', '')
flags.DEFINE_integer('environment_height', 84, '')
flags.DEFINE_integer('environment_width', 84, '')
flags.DEFINE_bool('use_gym', False, '')
flags.DEFINE_integer('replay_capacity', int(1e6), '')
flags.DEFINE_bool('compress_state', True, '')
flags.DEFINE_float('min_replay_capacity_fraction', 0.05, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_frames_per_episode', 108000, '')  # 30 mins.
flags.DEFINE_integer('num_action_repeats', 4, '')
flags.DEFINE_integer('num_stacked_frames', 4, '')
flags.DEFINE_float('exploration_epsilon_begin_value', 1., '')
flags.DEFINE_float('exploration_epsilon_end_value', 0.01, '')
flags.DEFINE_float('exploration_epsilon_decay_frame_fraction', 0.02, '')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, '')
flags.DEFINE_integer('target_network_update_period', int(4e4), '')
flags.DEFINE_bool('cramer', True, 'True for Cramer loss, false for QUANTILE LOSS')
flags.DEFINE_float('huber_param', 0., 'For QR-LOSS only, i.e. --nocramer')

flags.DEFINE_float('learning_rate', 0.00005, '')
flags.DEFINE_float('optimizer_epsilon', 0.01 / 32, '')
flags.DEFINE_float('additional_discount', 0.99, '')
flags.DEFINE_float('max_abs_reward', 1., '')
flags.DEFINE_float('max_global_grad_norm', 10., '')
flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
flags.DEFINE_integer('num_iterations', 200, '')
flags.DEFINE_integer('num_train_frames', int(1e6), '')  # Per iteration.
flags.DEFINE_integer('num_eval_frames', int(5e5), '')  # Per iteration.
flags.DEFINE_integer('learn_period', 16, '')
flags.DEFINE_string('results_csv_path', '/tmp/results.csv', '')

flags.DEFINE_integer('num_quantiles', 201, '')
flags.DEFINE_integer('n_nodes', 512, '')
flags.DEFINE_integer('n_layers', 1, '')

flags.DEFINE_bool('nc', False, '')
flags.DEFINE_bool('scale_grad', False, 'Scale gradient to make it equivalent to Cramer if not cramer, otherwise to QR ')



def main(argv):
  logging.info(FLAGS.flags_into_string())

  del argv
  logging.info('XX-XX-DQN on Atari on %s.',
               jax.lib.xla_bridge.get_backend().platform)
  random_state = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1))

  if FLAGS.results_csv_path:
    writer = parts.CsvWriter(FLAGS.results_csv_path)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    """Creates Atari environment."""
    env = gym_atari.GymAtari(
        FLAGS.environment_name, seed=random_state.randint(1, 2**32))
    return gym_atari.RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=random_state.randint(1, 2**32),
    )

  env = environment_builder()

  logging.info('Environment: %s', FLAGS.environment_name)
  logging.info('Action spec: %s', env.action_spec())
  logging.info('Observation spec: %s', env.observation_spec())
  num_actions = env.action_spec().num_values
  

  if FLAGS.nc:
      logging.info('NC network')
      network_fn = nc_atari_network(num_actions, FLAGS.num_quantiles,
                                       FLAGS.n_layers, FLAGS.n_nodes)
  else:
      logging.info('Standard QR network')
      network_fn = qr_atari_network(num_actions, FLAGS.num_quantiles)

  network = hk.transform(network_fn)


  def preprocessor_builder():
    return processors.atari(
        additional_discount=FLAGS.additional_discount,
        max_abs_reward=FLAGS.max_abs_reward,
        resize_shape=(FLAGS.environment_height, FLAGS.environment_width),
        num_action_repeats=FLAGS.num_action_repeats,
        num_pooled_frames=2,
        zero_discount_on_life_loss=True,
        num_stacked_frames=FLAGS.num_stacked_frames,
        grayscaling=True,
    )

  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(dm_env.TimeStep,
                                          sample_processed_timestep)
  sample_network_input = sample_processed_timestep.observation
  assert sample_network_input.shape == (FLAGS.environment_height,
                                        FLAGS.environment_width,
                                        FLAGS.num_stacked_frames)

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(FLAGS.min_replay_capacity_fraction * FLAGS.replay_capacity *
                  FLAGS.num_action_repeats),
      decay_steps=int(FLAGS.exploration_epsilon_decay_frame_fraction *
                      FLAGS.num_iterations * FLAGS.num_train_frames),
      begin_value=FLAGS.exploration_epsilon_begin_value,
      end_value=FLAGS.exploration_epsilon_end_value)

  if FLAGS.compress_state:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t))

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t))
  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  replay = replay_lib.TransitionReplay(FLAGS.replay_capacity, replay_structure,
                                       random_state, encoder, decoder)

  optimizer = optax.adam(
      learning_rate=FLAGS.learning_rate, eps=FLAGS.optimizer_epsilon)
  if FLAGS.max_global_grad_norm > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.max_global_grad_norm), optimizer)

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = QrDqn(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=network,
      #quantiles=quantiles,
      optimizer=optimizer,
      transition_accumulator=replay_lib.TransitionAccumulator(),
      replay=replay,
      batch_size=FLAGS.batch_size,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=FLAGS.min_replay_capacity_fraction,
      learn_period=FLAGS.learn_period,
      target_network_update_period=FLAGS.target_network_update_period,
      cramer=FLAGS.cramer,
      huber_param=FLAGS.huber_param,
      rng_key=train_rng_key,
	  num_quantiles=FLAGS.num_quantiles,
	  scale_grad=FLAGS.scale_grad
  )
  eval_agent = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network,
      exploration_epsilon=FLAGS.eval_exploration_epsilon,
      rng_key=eval_rng_key,
  )

  # Set up checkpointing.
  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.eval_agent = eval_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  while state.iteration <= FLAGS.num_iterations:
    # New environment for each iteration to allow for determinism if preempted.
    env = environment_builder()

    logging.info('Training iteration %d.', state.iteration)
    train_seq = parts.run_loop(train_agent, env, FLAGS.max_frames_per_episode)
    num_train_frames = 0 if state.iteration == 0 else FLAGS.num_train_frames
    train_seq_truncated = itertools.islice(train_seq, num_train_frames)
    train_stats = parts.generate_statistics(train_seq_truncated)

    logging.info('Evaluation iteration %d.', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env, FLAGS.max_frames_per_episode)
    eval_seq_truncated = itertools.islice(eval_seq, FLAGS.num_eval_frames)
    eval_stats = parts.generate_statistics(eval_seq_truncated)

    # Logging and checkpointing.
    human_normalized_score = atari_data.get_human_normalized_score(
        FLAGS.environment_name, eval_stats['episode_return'])
    capped_human_normalized_score = np.amin([1., human_normalized_score])
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * FLAGS.num_train_frames, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ('train_num_episodes', train_stats['num_episodes'], '%3d'),
        ('eval_frame_rate', eval_stats['step_rate'], '%4.0f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('normalized_return', human_normalized_score, '%.3f'),
        ('capped_normalized_return', capped_human_normalized_score, '%.3f'),
        ('human_gap', 1. - capped_human_normalized_score, '%.3f'),
    ]
    log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

  writer.close()


if __name__ == '__main__':
  config.update('jax_platform_name', 'gpu')  # Default to GPU.
  config.update('jax_numpy_rank_promotion', 'raise')
  config.config_with_absl()
  app.run(main)
