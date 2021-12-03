import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import OrbitalElements.orbitalPlotting as op
import tensorflow as tf
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Support.transformations import spherePines2cart
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

from environment import SafeModeEnv
from gravity_models import pinnGravityModel
from utils import collect_policy_checkpoints, load_policy
import pickle
from tf_agents.environments import wrappers

from visualization import visualize_returns_ci
def confidence_interval(returns, CI):
    #Confidence Interval x  +/-  0.95*(s/√n)
    std = np.std(returns)
    ci = CI*(std/len(returns))
    return ci


def execute_policy(env, policy, num_episodes):
    failures = []
    for k in range(num_episodes):
        print(k)
        time_step = env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
        failures.append(env.envs[0].failure_type)
    return failures

def eval_policy(policy_name, max_policy_idx, env, episodes):
    policy = load_policy(max_policy_idx, "Data/Policies/"+policy_name)
    num_episodes = 100
    failures = execute_policy(env, policy, episodes)

    with open("Data/Policies/" + policy_name + "/failures.data", 'wb') as f:
        pickle.dump(failures, f)    

def main():
    planet = Eros()
    pinn_model = pinnGravityModel("Data/DataFrames/eros_grav_model.data")   
    orig_env = SafeModeEnv(planet, pinn_model, reset_type='standard', random_seed=None)
    time_limit_env = wrappers.TimeLimit(orig_env, duration=1*60) # run for 600 steps at most
    env = BatchedPyEnvironment(envs=[time_limit_env])
    num_episodes = 100
    seed = 0
    np.random.seed(seed)
    policy_name = "PINN_Policy"
    max_policy = "0000110000"
    eval_policy(policy_name, max_policy, env, num_episodes)
    
    np.random.seed(seed)
    policy_name = "Poly_Policy"
    max_policy = "0000000200"
    eval_policy(policy_name, max_policy, env, num_episodes)

    np.random.seed(seed)
    policy_name = "Simple_Policy"
    max_policy = "0000270000"
    eval_policy(policy_name, max_policy, env, num_episodes)



if __name__ == "__main__":
    main()
