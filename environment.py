from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
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

import trimesh
import os
from scipy.integrate import solve_ivp
from gravity_model import xPrimeNN_Impulse
import pandas as pd
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import invert_projection, project_acceleration, sphere2cart, cart2sph
df = pd.read_pickle("Data/DataFrames/eros_grav_model.data")
_, gravity_model = load_config_and_model(df.iloc[0]['id'], df)

planet = Eros()
mu = planet.mu 
obj_file = planet.model_potatok
filename, file_extension = os.path.splitext(obj_file)
mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
proximity = trimesh.proximity.ProximityQuery(mesh)
    
def collision(t, y, grav_model, action):
    position_in_km = y[0:3]/1E3
    distance = proximity.signed_distance(position_in_km.reshape((1,3)))
    return distance
collision.terminal=True

def depart(t, y, grav_model, action):
    position_in_km = y[0:3]/1E3
    distance = Eros().radius*10/1E3 - np.abs(proximity.signed_distance(position_in_km.reshape((1,3)))[0])
    return distance
depart.terminal=True

def evolve_state(tVec, state, action):
  
    cart_pos = sphere2cart(state[0:3].reshape((1,3)))
    cart_vel = invert_projection(state[0:3].reshape((1,3)), state[3:6].reshape((1,3)))
    
    cart_state = np.squeeze(np.hstack((cart_pos, cart_vel)))
    sol = solve_ivp(xPrimeNN_Impulse,
                    tVec,
                    cart_state, 
                    rtol=1e-8, atol=1e-10, 
                    #t_eval=[tVec[1]], 
                    events=(collision, depart), 
                    args=(gravity_model, action)) 
    
    cart_state = sol.y[:,-1]
    sph_pos = cart2sph(cart_state[0:3].reshape((1,3)))
    sph_vel = project_acceleration(sph_pos, cart_state[3:6].reshape((1,3)))
    sph_state = np.squeeze(np.hstack((sph_pos, sph_vel)))
    # print(np.linalg.norm(sol.y[0:3,-1]))
    return np.squeeze(sph_state), (len(sol.t_events[0]) > 0 or len(sol.t_events[1] > 0))

class SafeModeEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(3,), minimum=-1, maximum=1, dtype=np.float32, name='action')
    self._observation_spec = array_spec.ArraySpec(
        shape=(6,), dtype=np.float32,  name='observation')
    self._state = array_spec.ArraySpec(
        shape=(6,), dtype=np.float32,  name='state')
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    r = np.random.uniform(Eros().radius*2, Eros().radius*3)
    theta = np.random.uniform(0, 360)
    phi = np.random.uniform(0,180)
    sph_pos = np.array([[r, theta, phi]])
    cart_pos = sphere2cart(sph_pos)
    self._state = np.squeeze(np.hstack((sph_pos,[[0,0,0]])))
    self._episode_ended = False
    return ts.restart(np.array(self._state, dtype=np.float32)) # TODO: looks like this should be observation

  def _step(self, action):

    if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
        return self.reset()

    # Make sure episodes don't go on forever.
    sp, terminal = evolve_state([0, 2*60], self._state, action) # episode interval is every 22 minutes
    if terminal:
        self._episode_ended = True
    else:
        self._state = sp

    if self._episode_ended:
        reward = -100 if terminal else 0.0
        return ts.termination(np.array(self._state, dtype=np.float32), reward)
    else:
        reward = 1 # For surviving
        reward -= self._state[0]/(Eros().radius*10) # for traveling far away from asteroid 
        reward -= np.linalg.norm(action) # For consuming fuel

        # Consider rewarding the S/C for getting close to the asteroid

        return ts.transition(
            np.array(self._state, dtype=np.float32), reward=reward, discount=0.99)


def main():
  env = SafeModeEnv()
  utils.validate_py_environment(env, episodes=5)

  # discrete_action_env = wrappers.ActionDiscretizeWrapper(env, num_actions=np.array([5,5,5]))
  run_stats_env = wrappers.RunStats(env)
  time_limit_env = wrappers.TimeLimit(run_stats_env, duration=60*60) # run for 1 hour at most
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
      action = tf.random.uniform([1,3], minval=-0.1, maxval=0.1, dtype=tf.float32).numpy()
      # action = tf.random.uniform([3], minval=0, maxval=4, dtype=tf.int32)
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