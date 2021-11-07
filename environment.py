from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from numba.misc.special import gdb
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from GravNN.CelestialBodies.Asteroids import Eros
from numba import njit, gdb_init
import trimesh
import os
from scipy.integrate import solve_ivp
from gravity_model import xPrimeNN_Impulse, simpleGravityModel
import pandas as pd
from OrbitalElements import oe
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sphPines, spherePines2cart
df = pd.read_pickle("Data/DataFrames/eros_grav_model.data")
_, gravity_model = load_config_and_model(df.iloc[0]['id'], df)


# gravity_model = simpleGravityModel(Eros().mu, Eros().radius)


planet = Eros()
R_ref = Eros().radius*10.0
M_ref = 540.0

a = Eros().radius
r = a
V_ref = np.sqrt(planet.mu*((2.0/r) - 0.0)) #escape velocity from Brill Sphere

mu = planet.mu 
obj_file = planet.model_potatok
filename, file_extension = os.path.splitext(obj_file)
mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
proximity = trimesh.proximity.ProximityQuery(mesh)
    
def collision(t, y, grav_model, action):
    cart_pos = y[0:3].reshape((1,3))
    cart_pos_in_km = cart_pos/1E3
    distance = proximity.signed_distance(cart_pos_in_km.reshape((1,3)))
    return distance
collision.terminal=True

def depart(t, y, grav_model, action):
    cart_pos = y[0:3].reshape((1,3))
    cart_pos_in_km = cart_pos/1E3
    distance = Eros().radius*10/1E3 - np.abs(proximity.signed_distance(cart_pos_in_km.reshape((1,3)))[0])
    return distance
depart.terminal=True

@njit()# cache=True causes segfaults ??
def get_integrator_state(norm_state, action):
    # gdb_init()
    state = unnormalize_state(norm_state)
    state_array = state[0:4].reshape((1,4))
    cart_pos = spherePines2cart(state_array).reshape((3,))
    cart_vel = state[4:7] + action.astype(np.float64) # add dV directly to velocity
    cart_state = np.hstack((cart_pos, cart_vel)).reshape((6,))
    dm = compute_fuel_cost(state[7], action)
    return cart_state, dm

@njit(cache=True)
def compute_fuel_cost(m_0, action):
    g = 9.80665 # constants drawn from BSK
    Isp = 226.7 # constants drawn from BSK
    v_e = Isp*g
    m_f = m_0 *np.exp(-np.linalg.norm(action)/v_e)
    return m_f

@njit(cache=True)
def generate_new_state(cart_state, m_f):
    sph_pines_pos = cart2sphPines(cart_state[0:3].reshape((1,3))).reshape((4,))
    cart_vel = cart_state[3:6]
    unorm_state = np.hstack((sph_pines_pos, cart_vel, np.array([m_f]))).reshape((8,))
    return normalize_state(unorm_state)


@njit(cache=True)
def normalize_state(state):
  r, s, t, u, vx, vy, vz, m = state
  norm_state = np.array([r/R_ref, s, t, u, vx/V_ref, vy/V_ref, vz/V_ref, m/M_ref])
  return norm_state

@njit(cache=True)
def unnormalize_state(norm_state):
  r_norm, s, t, u, vx_norm, vy_norm, vz_norm, m_norm = norm_state
  state = np.array([r_norm*R_ref, s, t, u, vx_norm*V_ref, vy_norm*V_ref, vz_norm*V_ref, m_norm*M_ref])
  return state


@njit(cache=True)
def flag_failure(t_events, m_f):
    # 500 kg is sc bus weight
    failure = False
    collided = len(t_events[0]) > 0
    departed = len(t_events[1]) > 0
    depleted = m_f < 500.0
    if collided:
      print("Failure: Collided")
      failure = True
    elif departed:
      print("Failure: Departed")
      failure = True
    elif depleted:
      print("Failure: Depleted")
      failure = True
    return failure

def evolve_state(tVec, state, action):
    cart_state, m_f = get_integrator_state(state, action)
    sol = solve_ivp(xPrimeNN_Impulse,
                    tVec,
                    cart_state, 
                    #rtol=1e-8, atol=1e-10, 
                    events=(collision, depart), 
                    args=(gravity_model, [0,0,0])) 

    cart_state = sol.y[:,-1]
    output_state = generate_new_state(cart_state, m_f)
    failure = flag_failure(sol.t_events, m_f)
    return output_state, failure

class SafeModeEnv(py_environment.PyEnvironment):

  def __init__(self, random_seed=None):
    if random_seed is not None:
      np.random.seed(random_seed)
    self.interval = 10*60
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(3,), minimum=-1, maximum=1, dtype=np.float32, name='action')
    self._observation_spec = array_spec.ArraySpec(
        shape=(8,), dtype=np.float32,  name='observation')
    self._state = array_spec.ArraySpec(
        shape=(8,), dtype=np.float32,  name='state')
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    r = np.random.uniform(Eros().radius*2, Eros().radius*3) 
    s = np.random.uniform(-1,1)
    t = np.random.uniform(-1,1)
    u = np.random.uniform(-1,1)
    
    sph_pos = np.array([r, s, t, u])
    sph_vel = np.array([0.0,0.0,0.0])
    bus_mass = 500.0 # kg
    fuel_mass = np.random.uniform(20, 40)
    sc_mass = bus_mass + fuel_mass

    state = np.hstack((sph_pos, sph_vel, sc_mass))#.reshape((1,-1))
    self._state = normalize_state(state)
    self._episode_ended = False
    return ts.restart(np.array(self._state, dtype=np.float32)) # TODO: looks like this should be observation

  def _step(self, action):

    if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
        return self.reset()

    # Make sure episodes don't go on forever.
    sp, terminal = evolve_state([0, self.interval], self._state, action) # episode interval is every 22 minutes
    if terminal:
        self._episode_ended = True
    else:
        dm = self._state[7] - sp[7] # Normalized fuel
        self._state = sp

    if self._episode_ended:
        reward = -100 if terminal else 0.0
        return ts.termination(np.array(self._state, dtype=np.float32), reward)
    else:
        reward = 1 # For surviving
        reward -= self._state[0]#/(Eros().radius*10) # for traveling far away from asteroid 
        reward -= dm * M_ref  # For consuming fuel 

        return ts.transition(
            np.array(self._state, dtype=np.float32), reward=reward, discount=0.99)


def main():
  env = SafeModeEnv()
  utils.validate_py_environment(env, episodes=1)

  # discrete_action_env = wrappers.ActionDiscretizeWrapper(env, num_actions=np.array([5,5,5]))
  run_stats_env = wrappers.RunStats(env)
  time_limit_env = wrappers.TimeLimit(run_stats_env, duration=1*60) # run for 600 steps at most
  tf_env = tf_py_environment.TFPyEnvironment(time_limit_env)
  # tf_env = env
  print("Action Spec: ", tf_env.action_spec())
  
  time_step = tf_env.reset()
  rewards = []
  steps = []
  num_episodes = 5

  for _ in range(num_episodes):
    episode_reward = 0
    episode_steps = 0
    while not time_step.is_last():
      action = tf.random.uniform([1,3], minval=-1, maxval=1, dtype=tf.float32).numpy()
      time_step = tf_env.step(action)
      episode_steps += 1
      episode_reward += time_step.reward#.numpy()
    rewards.append(episode_reward)
    steps.append(episode_steps)
    time_step = tf_env.reset()

  num_steps = np.sum(steps)
  avg_length = np.mean(steps)
  avg_reward = np.mean(rewards)

  print('num_episodes:', num_episodes, 'num_steps:', num_steps)
  print('avg_length', avg_length, 'avg_reward:', avg_reward)


if __name__ == "__main__":
  main()