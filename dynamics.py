import numpy as np

def xPrime(t, yVec, grav_model=None, thrust=[0,0,0]):
    u = yVec[3]
    v = yVec[4]
    w = yVec[5]

    acc = grav_model.generate_acceleration(yVec[0:3])

    uDot = acc[0,0] + thrust[0]
    vDot = acc[0,1] + thrust[1]
    wDot = acc[0,2] + thrust[2]

    xPrime = [u, v, w, uDot, vDot, wDot]

    return xPrime


