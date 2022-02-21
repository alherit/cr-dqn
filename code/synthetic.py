#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# parts of this code are copied from DQN_ZOO https://github.com/deepmind/dqn_zoo
# under the Apache License, Version 2.0 (the "License");


import sys

import typing

import haiku as hk
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np
import optax

from dqn_zoo import networks


import chex

import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42								 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net', help='nc,fc', default="fc")
parser.add_argument('--softplus',  action="store_true", default=False)
parser.add_argument('--njobs',  type=int, default=1)

parser.set_defaults(softplus=False)

parser.add_argument('--loss', help='qr_loss, cramer, wasserstein1', default="cramer")
parser.add_argument('--huber', type=float, default=0.)


parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--nodes_fc', type=int, default=45)
parser.add_argument('--nodes_nc', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--bs', type=int, default=32)


parser.add_argument('--N', type=int, default=12)

parser.add_argument('--loc1', type=float, default=-1.)
parser.add_argument('--loc2', type=float, default=1.)
parser.add_argument('--probloc1', type=float, default=2/3)
parser.add_argument('--probloc2', type=float, default=1/3)

parser.add_argument('--no-forcestep',  dest='force_step', action="store_false", default=True)
parser.add_argument('--no-mixture',  dest='mixture', action="store_false", default=True)


args = parser.parse_args()

plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 18

JIT = True
CONSTANT_INPUT = True
PLOT = True
num_actions = 1


FORCE_STEP = args.force_step
MIXTURE = args.mixture

batch_size = args.bs
NUM_QUANTILES = args.N  #number of quantiles 

N_LAYERS = args.layers
# this is to have similar number of params (2712 and 2702)
N_NODES_FC = args.nodes_fc
N_NODES_NC = args.nodes_nc


#network
NET = args.net #"fc",  "nc"
SOFTPLUS= args.softplus

HUBER = args.huber
LOSS = args.loss #cramer, qr_loss, wasserstein1
LR= args.lr

EPOCHS = args.epochs
TRIALS = args.trials


loc1 = args.loc1
loc2 = args.loc2
probloc1 = args.probloc1
probloc2 = args.probloc2

FLOAT_TYPE =  jnp.float32
UINT_TYPE = jnp.uint32

config.update("jax_debug_nans", True)


def force_step(dist_q_t,a,sort_key):
    if MIXTURE:
        return force_step2(dist_q_t,a,sort_key)
    else:
        step_location = loc1*probloc1 +  loc2*probloc2 #same as mixture
        return force_step_dirac(dist_q_t,a,step_location, sort_key)

def force_step2(dist_q_t,a,sort_key):
    sort_key, perturb_key = jax.random.split(sort_key, 2)
    r=jax.random.uniform(sort_key, (dist_q_t.shape[0],1))
    dist_qa_t = jnp.where(r > probloc1,
                           force_step_dirac(dist_q_t,a,loc2,perturb_key), 
                           force_step_dirac(dist_q_t,a,loc1,perturb_key))

    return dist_qa_t

def force_step_dirac(dist, a, R_t, perturb_key):
    return dist[:,:, a]*0. + R_t   


def force_staircase(dist_q_t, a):
    return dist_q_t[:,:, a]*0 + jnp.arange(loc1, loc2, (loc2-loc1)/NUM_QUANTILES)  






def nc_atari_network(num_actions: int, num_quantiles: int, 
                        n_layers: int, n_nodes: int ) -> networks.NetworkFn:

  def net_fn(inputs):
    """Function representing NC-QR-DQN Q-network."""

    if CONSTANT_INPUT:  
        torso_output = inputs
    else:
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

    if SOFTPLUS: # to avoid dying relu
        AMP = jax.nn.softplus(AMP)
    else:
        AMP = jax.nn.relu(AMP)
        
    Qprop = jax.nn.softmax(Rq, axis=1)

    Q = jnp.cumsum(Qprop,axis=1)


    Q *= AMP
   
    q_dist = Q +  Q0 
      
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return networks.QRNetworkOutputs(q_dist=q_dist, q_values=q_values)


  return net_fn


def dqn_value_head(num_actions: int, n_layers: int,
                   n_nodes: int , shared_bias: bool = False) -> networks.NetworkFn:
  """Regular DQN Q-value head with single hidden layer."""

  last_layer = networks.linear_with_shared_bias if shared_bias else networks.linear

  def net_fn(inputs):
    """Function representing value head for a DQN Q-network."""
    model = []
    for i in range(n_layers):
        model += [networks.linear(n_nodes),jax.nn.relu]
    network = hk.Sequential(model +[last_layer(num_actions),])
    return network(inputs)

  return net_fn



def qr_atari_network(num_actions: int, num_quantiles: int, 
                        n_layers: int, n_nodes: int ) -> networks.NetworkFn:
  """QR-DQN network, expects `uint8` input."""


  def net_fn(inputs):
    """Function representing QR-DQN Q-network."""
    if CONSTANT_INPUT:  
        model = [dqn_value_head(num_quantiles * num_actions, n_layers, n_nodes),]
    else:
        model = [networks.dqn_torso(),networks.dqn_value_head(num_quantiles * num_actions),]
   

    network = hk.Sequential(model)
    network_output = network(inputs)
    q_dist = jnp.reshape(network_output, (-1, num_quantiles, num_actions))
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return networks.QRNetworkOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn



from rlax._src import clipping
Array = chex.Array
Numeric = chex.Numeric


def cramer_dist(
    dist_src: Array,
    dist_target: Array,
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
  return jnp.sum(integs) * n / 2.  ## make it equivalent to QR, with respect to ADAM's epsilon


    
    
    
def _quantile_regression_loss(
    dist_src: Array,
    tau_src: Array,
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

  # Average over target-j (-1 last dimension), sum over src-i (first dimension).
  return jnp.sum(jnp.mean(loss, axis=-1))




def wasserstein1(dist_src: jnp.array, dist_target: jnp.array):
    # divide by num_quantiles since it is the height of each rectangle
    qdiff = jnp.abs(jax.lax.sort(dist_src) - jax.lax.sort(dist_target))/NUM_QUANTILES
    return jnp.sum(qdiff)    




def _loss(
    dist_src: Numeric,
    dist_target: Numeric
) -> Numeric:
  """Compute Cramer loss between two discrete quantile-valued distributions.
  Args:
    dist_src: source probability distribution.
    dist_target: target probability distribution.
  Returns:
    Cramer loss.
  """
  chex.assert_rank([dist_src, dist_target], 1)
  chex.assert_type([dist_src, dist_target], float)


  if LOSS=="wasserstein1":
    res = wasserstein1(dist_src,dist_target)
  elif LOSS=="qr_loss":
    tau_src = (jnp.arange(0, NUM_QUANTILES) + 0.5) / float(NUM_QUANTILES)
    res = _quantile_regression_loss(dist_src,tau_src,dist_target,huber_param=HUBER)
  elif LOSS=="cramer": 
    res =  cramer_dist(dist_src,dist_target)
  else:
    exit(1)

 
  return jnp.squeeze(res)





_batch_loss = jax.vmap(
    _loss, in_axes=(0, 0)
    )






if NET=="nc":
    network_fn = nc_atari_network(num_actions, NUM_QUANTILES, N_LAYERS, N_NODES_NC)
elif NET=="fc":
    network_fn = qr_atari_network(num_actions, NUM_QUANTILES, N_LAYERS, N_NODES_FC)
else:
    exit(1)
    







def trial(j):
    print("TRIAL: ",j)


    seed = j
    print("seed: ", seed)
    random_state = np.random.RandomState(seed)

    sample_network_input = jnp.ones((batch_size,1))


    method = NET
    if NET=="nc" and SOFTPLUS:
        method += "_softplus"
    method += "_"+LOSS
    if LOSS=="qr_loss":
        method += "_k" + str(HUBER)    
    
    fname = "output_staircase_"+ method

    fname += "_" + str(j) + ".txt"
    f = open(fname,'w')

    
    network = hk.transform(network_fn)

    def loss_fn(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      # Compute Q value distributions.
      rng_key, online_key, target_key = jax.random.split(rng_key, 3)
      dist_q_tm1 = network.apply(online_params, online_key,
                                 transitions).q_dist
      target = network.apply(target_params, target_key,
                                      transitions)
      
      dist_q_t =  target.q_dist
      
      dist_q_t = jax.lax.stop_gradient(dist_q_t)
     
        
    
      # Only update action 0.
      a = 0
      dist_qa_tm1 = dist_q_tm1[:, :, a]
    
    
      # Select target for action 0
    
      
      if FORCE_STEP:
        _, sort_key = jax.random.split(rng_key, 2)
        dist_qa_t = force_step(dist_q_t,a, sort_key) 
      else:
        dist_qa_t = force_staircase(dist_q_t,a) 
        #dist_qa_t = dist_q_t[:, :, a] 
            
    
    
      losses = _batch_loss(
        dist_qa_tm1,
        dist_qa_t
      )
      
    
      loss = jnp.mean(losses)
      
      return loss

    def update(rng_key, opt_state, online_params, target_params, transitions):
      rng_key, update_key = jax.random.split(rng_key)
    
        
      loss,d_loss_d_params = jax.value_and_grad(loss_fn)(online_params,
                                                        target_params, transitions, update_key)
          
    
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params, loss
    
    
    
    if JIT:
        update = jax.jit(update)

    

    rng_key = jax.random.PRNGKey(
          random_state.randint(-sys.maxsize - 1, sys.maxsize + 1))
    
    # Initialize network parameters and optimizer.
    rng_key, network_rng_key = jax.random.split(rng_key)
    online_params = network.init(network_rng_key,
                                           sample_network_input)
    
    
    rng_key, target_rng_key = jax.random.split(rng_key)
    target_params = network.init(target_rng_key,
                                           sample_network_input)

    print("Number of trainable params: ", hk.data_structures.tree_size(online_params))

    
   
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(online_params)
    
    
    for i in range(EPOCHS):
    
    
      rng_key, opt_state, online_params, loss  = update(
            rng_key,
            opt_state,
            online_params,
            target_params,
            sample_network_input
            )
          
    
    
      learned_out = network.apply(online_params, rng_key, sample_network_input)      
      target_out = network.apply(target_params, rng_key, sample_network_input)      
      dist_learned = learned_out.q_dist
      dist_target = target_out.q_dist
      
      
      a = 0
    
        
      if FORCE_STEP:
         rng_key , sort_key = jax.random.split(rng_key, 2)
         dist_target = force_step(dist_target,a, sort_key)
      else:
         #dist_target = dist_target[:, :, a]
         dist_target = force_staircase(dist_target,a) 
    
        
   
      if True:
          out_str =  "TRIAL: " + str(j) + " i:"+ str(i) + " loss: " + str(loss) +\
          " expectation f1: " + str(float(learned_out.q_values[0,0])) +\
          " Q1:"+ str(list(dist_learned[:,:,a][0,:])) + " Q2:"+ str(list(dist_target[0,:]))
          print(out_str,file=f)
      


    Q1 = jax.lax.sort(dist_learned[:,:,a], dimension=1)
    Q1 = list(Q1[0,:])
    Q2 = jax.lax.sort(dist_target, dimension=1)
    Q2 = list(Q2[0,:])
    

    f.close()

    
    return Q1,Q2



from joblib import Parallel, delayed
res =Parallel(n_jobs=args.njobs)(delayed(trial)(j) for j in range(TRIALS))


Ts = np.cumsum(np.ones(NUM_QUANTILES)/NUM_QUANTILES)



   

    
wasser1 = []

for Q1,Q2 in res:

    if FORCE_STEP and MIXTURE:
        Q2 = [loc1]*round(NUM_QUANTILES*probloc1) +  [loc2]*round(NUM_QUANTILES*probloc2)

    wasser1.append(float(wasserstein1(np.array(Q1),np.array(Q2))))

    #extend the lines by this
    ext = abs(loc2-loc1)*.1

    plTs = [0.,0.]+ [val for val in Ts for _ in (0, 1)]
    plQ1 = [min(loc1,loc2)-ext] + [val for val in Q1 for _ in (0, 1)] + [max(loc1,loc2)+ext]
    plQ2 = [min(loc1,loc2)-ext] + [val for val in Q2 for _ in (0, 1)] + [max(loc1,loc2)+ext]
    
    plt.plot(plQ1,plTs,color="b",alpha=0.4)
    plt.plot(plQ2,plTs,color="r")


method = NET
if NET=="nc" and SOFTPLUS:
    method += "_softplus"
method += " "+LOSS
if LOSS=="qr_loss":
    method += " κ=" + str(HUBER)    

method = method.replace("fc","FC").replace("nc","NC").replace("cramer","Cramér").replace("qr_loss","QR loss")
method = method.replace("wasserstein1","1-Wasserstein loss")

plt.title(method + "  d₁=" + '%.4f' % np.mean(wasser1) + "±" + '%.4f' % np.std(wasser1))

plotfname = NET
if NET=="nc" and SOFTPLUS:
    plotfname+="_softplus"
plotfname += "_"+ LOSS 
if LOSS=="qr_loss":
    method += "_k" + str(HUBER)    
plotfname+=str(HUBER)+'.pdf'
plt.savefig(plotfname, bbox_inches = 'tight')
#plt.show()
        
