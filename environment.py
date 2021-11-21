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
from dynamics import *
import pandas as pd
from OrbitalElements import oe
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sphPines, spherePines2cart, sphere2cart
from gravity_models import pinnGravityModel, polyhedralGravityModel
from OrbitalElements.coordinate_transforms import oe2cart_tf

@njit()# cache=True causes segfaults ??
def get_integrator_state(norm_state, action, R_ref, V_ref, M_ref):
    # gdb_init()
    state = unnormalize_state(norm_state, R_ref, V_ref, M_ref)
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
def generate_new_state(cart_state, m_f, R_ref, V_ref, M_ref):
    sph_pines_pos = cart2sphPines(cart_state[0:3].reshape((1,3))).reshape((4,))
    cart_vel = cart_state[3:6]
    unorm_state = np.hstack((sph_pines_pos, cart_vel, np.array([m_f]))).reshape((8,))
    return normalize_state(unorm_state, R_ref, V_ref, M_ref)

@njit(cache=True)
def normalize_state(state, R_ref, V_ref, M_ref):
  r, s, t, u, vx, vy, vz, m = state
  norm_state = np.array([r/R_ref, s, t, u, vx/V_ref, vy/V_ref, vz/V_ref, m/M_ref])
  return norm_state

@njit(cache=True)
def unnormalize_state(norm_state, R_ref, V_ref, M_ref):
  r_norm, s, t, u, vx_norm, vy_norm, vz_norm, m_norm = norm_state
  state = np.array([r_norm*R_ref, s, t, u, vx_norm*V_ref, vy_norm*V_ref, vz_norm*V_ref, m_norm*M_ref])
  return state

@njit(cache=True)
def flag_failure(t_events, m_f):
    # 500 kg is sc bus weight
    failure = False
    failure_type = "Succeeded"
    collided = len(t_events[0]) > 0
    departed = len(t_events[1]) > 0
    depleted = m_f < 500.0
    if collided:
      print("Failure: Collided")
      failure = True
      failure_type = "Collided"
    elif departed:
      print("Failure: Departed")
      failure = True
      failure_type = "Departed"
    elif depleted:
      print("Failure: Depleted")
      failure = True
      failure_type = 'Depleted'
    return failure, failure_type

class SafeModeEnv(py_environment.PyEnvironment):
  def __init__(self, planet, gravity_model, reset_type='standard', random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    self.interval = 10*60
    self.gravity_model = gravity_model
    self.reset_type = reset_type
    self.planet = planet
    self.failure_type = None

    # Configure the shape mesh and reference velocity and radius
    obj_file = planet.model_potatok
    filename, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    self.proximity = trimesh.proximity.ProximityQuery(mesh)

    self.M_ref = 540.0
    self.R_ref = planet.radius*3.0#10.0
    self.V_ref = np.sqrt(planet.mu*((2.0/planet.radius) - 0.0)) #escape velocity from Brill Sphere

    setattr(SafeModeEnv.collision, "terminal", True)
    setattr(SafeModeEnv.depart, "terminal", True)

    # setattr(SafeModeEnv.collision, "count", 0)
    # setattr(SafeModeEnv.depart, "count", 0)

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
    if self.reset_type == "standard":
      # Attempting to avoid sampling within the asteroid sphere
      r = np.random.uniform(self.planet.radius + 10000, self.R_ref) 
      theta = np.random.uniform(0, 360.0)
      phi = np.random.uniform(0, 180.0)
      pos_sph = np.array([[r, theta, phi]])
      pos_cart = sphere2cart(pos_sph)

      s = pos_cart[0,0] / r
      t = pos_cart[0,1] / r
      u = pos_cart[0,2] / r
      
      sph_pos = np.array([r, s, t, u])
      sph_vel = np.array([0.0,0.0,0.0]) + np.random.uniform(-1,1, size=(3,))
      bus_mass = 500.0 # kg
      fuel_mass = np.random.uniform(20, 40)
      sc_mass = bus_mass + fuel_mass

      state = np.hstack((sph_pos, sph_vel, sc_mass))#.reshape((1,-1))
      self._state = normalize_state(state, self.R_ref, self.V_ref, self.M_ref)
      self._episode_ended = False
      return ts.restart(np.array(self._state, dtype=np.float32)) # TODO: looks like this should be observation

    elif self.reset_type == 'orbiting':
      a = np.random.uniform(self.planet.radius, self.R_ref)
      e_1 = 1. - self.planet.radius/a
      e_2 = self.R_ref/a - 1.
      e = np.random.uniform(0, np.min([e_1, e_2]))

      trad_OE = np.array([[a, 
                           e, 
                           np.random.uniform(0.0, np.pi),
                           np.random.uniform(0.0, 2*np.pi),
                           np.random.uniform(0.0, 2*np.pi),
                           np.random.uniform(0.0, 2*np.pi)]]) 

      rVec, vVec = oe2cart_tf(trad_OE, self.planet.mu)

      r = np.linalg.norm(rVec.numpy())

      s = rVec.numpy()[0,0] / r
      t = rVec.numpy()[0,1] / r
      u = rVec.numpy()[0,2] / r
      
      sph_pos = np.array([r, s, t, u])
      sph_vel = vVec.numpy()[0]
      bus_mass = 500.0 # kg
      fuel_mass = np.random.uniform(20, 40)
      sc_mass = bus_mass + fuel_mass

      state = np.hstack((sph_pos, sph_vel, sc_mass))#.reshape((1,-1))
      self._state = normalize_state(state, self.R_ref, self.V_ref, self.M_ref)
      self._episode_ended = False
      return ts.restart(np.array(self._state, dtype=np.float32)) # TODO: looks like this should be observation
    else:
      return NotImplementedError()


  def _step(self, action):

    if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
        return self.reset()

    # Make sure episodes don't go on forever.
    sp, terminal = self.evolve_state([0, self.interval], self._state, action) # episode interval is every 22 minutes
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
        reward -= self._state[0] # for traveling far away from asteroid 
        reward -= dm * self.M_ref  # For consuming fuel 

        return ts.transition(
            np.array(self._state, dtype=np.float32), reward=reward, discount=0.99)

  def evolve_state(self, tVec, state, action):
      cart_state, m_f = get_integrator_state(state, action, self.R_ref, self.V_ref, self.M_ref)
      sol = solve_ivp(xPrime,
                      tVec,
                      cart_state, 
                      #rtol=1e-8, atol=1e-10, 
                      events=(self.collision, self.depart), 
                      args=(self.gravity_model, [0,0,0])) 

      cart_state = sol.y[:,-1]
      output_state = generate_new_state(cart_state, m_f, self.R_ref, self.V_ref, self.M_ref)
      failure, failure_type = flag_failure(sol.t_events, m_f)
      self.failure_type = failure_type
      return output_state, failure

  def collision(self, t, y, grav_model, action):
      cart_pos = y[0:3].reshape((1,3))
      cart_pos_in_km = cart_pos/1E3
      distance = self.proximity.signed_distance(cart_pos_in_km.reshape((1,3)))
      return distance

  def depart(self, t, y, grav_model, action):
      cart_pos = y[0:3].reshape((1,3))
      cart_pos_in_km = cart_pos/1E3
      distance = self.R_ref/1E3 - np.abs(self.proximity.signed_distance(cart_pos_in_km.reshape((1,3)))[0])
      return distance



def main():

  planet = Eros()
  gravity_model = pinnGravityModel("Data/DataFrames/eros_grav_model.data")
  gravity_model = polyhedralGravityModel(planet, planet.obj_200k)
  gravity_model = polyhedralGravityModel(planet, planet.model_7790)

  env = SafeModeEnv(planet, gravity_model, reset_type='orbiting')
  utils.validate_py_environment(env, episodes=1)

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