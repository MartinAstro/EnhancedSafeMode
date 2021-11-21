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

from visualization import visualize_returns_ci
def confidence_interval(returns, CI):
    #Confidence Interval x  +/-  0.95*(s/√n)
    std = np.std(returns)
    ci = CI*(std/len(returns))
    return ci


def execute_policy(env, policy, num_episodes):
    failures = []
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
        failures.append(env.envs[0].failure_type)
    return failures
    
def main():

    planet = Eros()
    pinn_model = pinnGravityModel("Data/DataFrames/eros_grav_model.data")   
    env = BatchedPyEnvironment(envs=[SafeModeEnv(planet, pinn_model, reset_type='standard', random_seed=None)])
    policy_name = "PINN_Policy"
    max_policy = "0000000150"

    policy_name = "Poly_Policy"
    max_policy = "0000000050"

    policy_name = "Simple_Policy"
    max_policy = "0000001000"

    policy = load_policy(max_policy, "Data/Policies/"+policy_name)
    num_episodes = 4
    failures = execute_policy(env, policy, num_episodes)

    with open("Data/Policies/" + policy_name + "/failures.data", 'wb') as f:
        pickle.dump(failures, f)


if __name__ == "__main__":
    main()
