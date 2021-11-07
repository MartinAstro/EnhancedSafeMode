import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import OrbitalElements.orbitalPlotting as op
import tensorflow as tf
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Support.transformations import spherePines2cart
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
import copy
from environment import R_ref, SafeModeEnv
from utils import load_policy
from GravNN.Visualization.VisualizationBase import VisualizationBase



def visualize_returns(num_iterations, eval_interval, returns):
    # Visualization
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim()
    plt.show()

def visualize_returns_ci(data):
    vis = VisualizationBase()
    vis.newFig()
    steps = data['steps']
    returns = data['avg_returns']
    conf_intervals = data['confidence_intervals']
    plt.plot(steps, returns, color='blue')
    plt.fill_between(steps, 
                    (returns-conf_intervals), 
                    (returns+conf_intervals), 
                    color='blue', alpha=0.5)
    
def main_mod():
    step = '0000021000'
    step = '0000045000'
    policy = load_policy(step)
    env = BatchedPyEnvironment(envs=[SafeModeEnv(random_seed=None)])

    time_step = env.reset()
    i = 0

    rVec = []
    tVec = []
    total_reward = 0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        y = time_step.observation
        r, s, t, u = y[0,0:4]
        rf = np.array(spherePines2cart(np.array([r*R_ref, s, t, u]).reshape((1,4)))).squeeze()
        rVec.append(rf)
        tVec.append(env.envs[0].interval*i)
        i += 1
        total_reward += time_step.reward
        if i % 1 == 0:
            print(i)

    rVec = np.array(rVec).T
    tVec = np.array(tVec)
    print("Max Time: %d \t Return: %f" % (np.max(tVec), total_reward))
    op.plot3d(tVec, rVec, None, '3d', show=True, obj_file=Eros().model_potatok, save=False)

    return 
if __name__ == "__main__":
    main_mod()
