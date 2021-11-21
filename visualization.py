import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import OrbitalElements.orbitalPlotting as op
import tensorflow as tf
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Support.transformations import sphere2cart, spherePines2cart
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
import copy
from environment import SafeModeEnv
from utils import load_policy
from GravNN.Visualization.VisualizationBase import VisualizationBase



def visualize_returns(steps, returns):
    # Visualization
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
    
def plot_gravity_field():
    import pandas as pd
    from GravNN.Networks.Model import load_config_and_model
    df = pd.read_pickle("Data/DataFrames/eros_grav_model.data")
    config, gravity_model = load_config_and_model(df.iloc[0]['id'], df)

    vis = VisualizationBase()
    vis.newFig()
    rVec = np.linspace(Eros().radius, Eros().radius*10,1000)
    aVec = []
    pVec = []
    for r in rVec:
        cart_pos = sphere2cart(np.array([r, 0, 0]).reshape((1,3))).reshape((3,))
        pVec.append(cart_pos)
    X = np.array(pVec).reshape((-1,3)).astype(np.float32)
    aVec = gravity_model.generate_acceleration(X).numpy()
    a_mag = np.linalg.norm(aVec, axis=1)
    plt.plot(rVec, a_mag)
    plt.show()

def main_mod():
    step = '0000021000'
    step = '0000045000'
    step = '0000300000'
    step = '0000100000'
    load_dir = os.path.expanduser('~') + "/Desktop/BestPolicy/"
    policy = load_policy(step, load_dir=load_dir)
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
    # plot_gravity_field()
