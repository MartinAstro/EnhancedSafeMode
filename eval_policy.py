import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import OrbitalElements.orbitalPlotting as op
import tensorflow as tf
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Support.transformations import spherePines2cart
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

from environment import R_ref, SafeModeEnv
from utils import collect_policy_checkpoints, load_policy
import pickle

from visualization import visualize_returns_ci
def confidence_interval(returns, CI):
    #Confidence Interval x  +/-  0.95*(s/√n)
    std = np.std(returns)
    ci = CI*(std/len(returns))
    return ci


def evaluate_policy(env, policy, num_episodes):
    rewardVec = []
    for _ in range(num_episodes):
        time_step = env.reset()
        i = 0   
        total_reward = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            total_reward += time_step.reward
            i += 1
        rewardVec.append(total_reward)

    avg_return = np.mean(rewardVec)
    ci = confidence_interval(rewardVec, 0.95)
    print("Return: %.2f +/- %.2f" % (avg_return, ci))
    return avg_return, ci

def load_evaluation_data():
    try:
        with open("Data/returns.data", 'rb') as f:
            data = pickle.load(f)
    except:
        data = {'steps' : np.array([]), 
                 'avg_returns' : np.array([]),
                 'confidence_intervals' : np.array([])}
    return data
    
def main():
    env = BatchedPyEnvironment(envs=[SafeModeEnv(random_seed=None)])

    num_episodes = 10
    policy_checkpoints = collect_policy_checkpoints()

    data = load_evaluation_data()
    results = []
    for policy_ckpt in policy_checkpoints:
        if int(policy_ckpt) in data['steps']:
            continue
        print(policy_ckpt)
        policy = load_policy(policy_ckpt)
        avg_return, ci = evaluate_policy(env, policy, num_episodes)
        results.append([int(policy_ckpt), avg_return, ci])
    
    results = np.array(results).reshape((-1,3))
    data = {
        "steps" : np.append(data['steps'],results[:,0]),
        "avg_returns" : np.append(data['avg_returns'],results[:,1]),
        'confidence_intervals' : np.append(data['confidence_intervals'],results[:,2])
    }
    visualize_returns_ci(data)

    with open("Data/returns.data", 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
