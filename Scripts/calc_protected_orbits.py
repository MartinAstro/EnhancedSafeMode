import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import OrbitalElements.orbitalPlotting as op
import tensorflow as tf
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Support.transformations import spherePines2cart
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

from Environments.ESM_MDP import SafeModeEnv
from gravity_models import PINNGravityModel
from utils import load_policy, save_policy
import pickle
from tf_agents.environments import wrappers


def get_trajectory(env, policy):
    time_step = env.reset()
    i = 0

    rVec = []
    tVec = []
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        y = time_step.observation
        r, s, t, u = y[0,0:4]
        rf = np.array(spherePines2cart(np.array([r*env.envs[0].R_ref, s, t, u]).reshape((1,4)))).squeeze()
        rVec.append(rf)
        tVec.append(env.envs[0].interval*i)
        i += 1

    rVec = np.array(rVec).T
    tVec = np.array(tVec)
    return rVec, tVec


def save_trajectory(i, policy_name, max_policy_idx, env):
    policy = load_policy(max_policy_idx, "Data/Policies/"+policy_name)
    rVec, tVec = get_trajectory(env, policy)
    with open("Data/Trajectories/"+policy_name+"_trajectory_"+str(i)+".data", 'wb') as f:
        pickle.dump(tVec, f)
        pickle.dump(rVec, f)


def main():
    seed = 13 # 13 is best
    planet = Eros()
    pinn_model = PINNGravityModel("Data/DataFrames/eros_grav_model.data")   
    orig_env = SafeModeEnv(planet, pinn_model, reset_type='standard', random_seed=None)
    time_limit_env = wrappers.TimeLimit(orig_env, duration=1*60*5) # run for 600 steps at most
    env = BatchedPyEnvironment(envs=[time_limit_env])
    np.random.seed(seed)
    for i in range(3):    
        policy_name = "PINN_Policy"
        max_policy = "0000110000"
        save_trajectory(i, policy_name, max_policy, env)
        
    np.random.seed(seed)
    for i in range(3):    
        policy_name = "Poly_Policy"
        max_policy = "0000000200"
        save_trajectory(i, policy_name, max_policy, env)

    np.random.seed(seed)
    for i in range(3):    
        policy_name = "Simple_Policy"
        max_policy = "0000270000"
        save_trajectory(i, policy_name, max_policy, env)
    

if __name__ == "__main__":
    main()
