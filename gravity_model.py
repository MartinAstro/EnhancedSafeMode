from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Asteroids import Eros
from OrbitalElements.numSolver import solveOrbitODE
from OrbitalElements import utils
from OrbitalElements import oe
from OrbitalElements.perturbations import xPrimeNN
import OrbitalElements.orbitalPlotting as op
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
from GravNN.Support.ProgressBar import ProgressBar
import trimesh
import os
def time_network():
    df = pd.read_pickle("Data/DataFrames/eros_grav_model.data")
    _, gravity_model = load_config_and_model(df.iloc[0]['id'], df)

    planet = Eros()
    mu = planet.mu 

    a = planet.radius*3
    OE = [a, 0.1, np.pi/4.0, 0.0, 0.0, 0.0]
    n = oe.compute_n(OE[0], mu)
    T = oe.compute_T(n)

    r0, v0 = utils.computeTwoBodyPosVel(OE, [0.], mu)
    action = [0,0,0] 

    tVec = np.arange(0, 3*T, 3*T/100)
    rVec = np.zeros((3, len(tVec)))
    vVec = np.zeros((3, len(tVec)))

    rVec[:,0] = r0
    vVec[:,0] = v0
    r0 = np.array(r0).reshape((1,3)).astype(np.float32)
    r0 = tf.cast(r0, tf.float32)
    import timeit
    def wrapper(r0, gravity_model):
        # import numpy as np
        def timing_function():
            # X = np.array(r0).reshape((1,3)).astype(np.float32)
            gravity_model.generate_acceleration(r0).numpy()
        return timing_function
    wrapped = wrapper(r0, gravity_model)
    number = 3000
    time = np.array(timeit.repeat(wrapped, number=number, repeat=10))
    print((np.mean(time/number), np.std(time/number)))
    return



def xPrimeNN_Impulse(t, yVec, NN=None, thrust=[0,0,0]):
    # Get velocities
    u = yVec[3]
    v = yVec[4]
    w = yVec[5]

    # Generate gravitational acceleration
    X = np.array(yVec[0:3]).reshape((1,3)).astype(np.float32)
    acc = NN.generate_acceleration(X).numpy()

    uDot = acc[0,0] + thrust[0]
    vDot = acc[0,1] + thrust[1]
    wDot = acc[0,2] + thrust[2]

    xPrime = [u, v, w, uDot, vDot, wDot]

    return xPrime



def main_mod():
    df = pd.read_pickle("Data/DataFrames/eros_grav_model.data")
    _, gravity_model = load_config_and_model(df.iloc[0]['id'], df)

    planet = Eros()
    mu = planet.mu 
    obj_file = planet.model_potatok
    filename, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    proximity = trimesh.proximity.ProximityQuery(mesh)
        
    def event(t, y, grav_model, action):
        distance = proximity.signed_distance(y[0:3].reshape((1,3)))
        return distance
    
    event.terminal = True

    a = planet.radius*3
    OE = [a, 0.1, np.pi/4.0, 0.0, 0.0, 0.0]
    n = oe.compute_n(OE[0], mu)
    T = oe.compute_T(n)

    r0, v0 = utils.computeTwoBodyPosVel(OE, [0.], mu)
    v0 = np.zeros((3,1))
    action = np.array([0,0,0])

    points_per_orbit = 100
    orbits = 1
    tVec = np.arange(0, orbits*T, T/points_per_orbit)
    rVec = np.zeros((3, len(tVec)))
    vVec = np.zeros((3, len(tVec)))

    rVec[:,0] = r0[:,0]
    vVec[:,0] = v0[:,0]

    pbar = ProgressBar(len(tVec)-1, enable=True)
    for i in range(1,len(tVec)):
        y = np.vstack((r0, v0)).T.squeeze()
        sol = solve_ivp(xPrimeNN_Impulse,[
                        tVec[i-1], tVec[i]], 
                        y, 
                        rtol=1e-8, atol=1e-10, 
                        t_eval=[tVec[i]], 
                        events=(event), 
                        args=(gravity_model, action)) 

        # If trajectory intersects the surface -- stop simulation
        if len(sol.t_events) > 0:
            break

        rf = np.array(sol.y[0:3]).squeeze()
        vf = np.array(sol.y[3:6]).squeeze()

        rVec[:,i] = rf
        vVec[:,i] = vf

        r0 = np.reshape(rf, (3,1))
        v0 = np.reshape(vf, (3,1))
        pbar.update(i)
    
    pbar.close()
    tVec = tVec[:i]
    op.plot3d(tVec, rVec[:,:len(tVec)], vVec[:,:len(tVec)], '3d', show=True, obj_file=planet.model_potatok)

    return

if __name__ == "__main__":
    main_mod()
    # time_network()