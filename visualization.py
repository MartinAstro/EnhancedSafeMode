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
    policy_dir = os.path.join(tempdir, 'policies/checkpoints/policy_checkpoint_0000200000')
    policy = tf.saved_model.load(policy_dir)

    env = BatchedPyEnvironment(envs=[SafeModeEnv()])

    time_step = env.reset()
    i = 0

    rVec = []
    tVec = []
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        y = time_step.observation
        pines_pos = y[0,0:4].reshape((1,4))
        rf = np.array(spherePines2cart(pines_pos)).squeeze()
        rVec.append(rf)
        tVec.append(env.interval*i)
        i += 1

    rVec = np.array(rVec).T
    tVec = np.array(tVec)
    print(np.max(tVec))
    op.plot3d(tVec, rVec, None, '3d', show=True, obj_file=Eros().model_potatok, save=False)

if __name__ == "__main__":
    main_mod()
