import os
import tempfile

import numpy as np
from GravNN.Support.transformations import sphere2cart
import OrbitalElements.orbitalPlotting as op
import pandas as pd
import tensorflow as tf
import trimesh
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.ProgressBar import ProgressBar
from OrbitalElements import oe, utils
from scipy.integrate import solve_ivp
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

from environment import SafeModeEnv
from gravity_model import xPrimeNN_Impulse
import matplotlib.pyplot as plt

def visualize_returns(num_iterations, eval_interval, returns):
    # Visualization
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim()
    plt.show()



def main_mod():
    tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())

    # Load policy
    policy_dir = os.path.join(tempdir, 'policies/policy')
    saved_policy = tf.saved_model.load(policy_dir)

    env = BatchedPyEnvironment(envs=[SafeModeEnv()])

    df = pd.read_pickle("Data/DataFrames/eros_grav_model.data")
    _, gravity_model = load_config_and_model(df.iloc[0]['id'], df)

    planet = Eros()
    mu = planet.mu 


    a = planet.radius*3
    OE = [a, 0.1, np.pi/4.0, 0.0, 0.0, 0.0]
    n = oe.compute_n(OE[0], mu)
    T = oe.compute_T(n)


    points_per_orbit = 100
    orbits = 1
    tVec = np.arange(0, orbits*T, T/points_per_orbit)
    rVec = np.zeros((3, len(tVec)))
    vVec = np.zeros((3, len(tVec)))

    time_step = env.reset()
    i = 0
    while not time_step.is_last():
        action_step = saved_policy.action(time_step)
        time_step = env.step(action_step.action)

        y = time_step.observation
        rf = np.array(sphere2cart(y[0,0:3].reshape((1,3)))).squeeze()

        rVec[:,i] = rf

        i += 1

    tVec = tVec[:i]
    op.plot3d(tVec, rVec[:,:len(tVec)], vVec[:,:len(tVec)], '3d', show=True, obj_file=planet.model_potatok)

if __name__ == "__main__":
    main_mod()
