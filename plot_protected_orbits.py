import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from GravNN.Visualization.VisualizationBase import VisualizationBase
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import OrbitalElements.orbitalPlotting as op
from GravNN.CelestialBodies.Asteroids import Eros
import matplotlib.pyplot as plt
import os
def read_trajectory(file):
    with open(file, 'rb') as f:
        tVec = pickle.load(f)
        rVec = pickle.load(f)
    return (tVec, rVec)

def main():

    colors = [plt.cm.spring, plt.cm.autumn, plt.cm.summer]
    colors = [plt.cm.Blues, plt.cm.Greens, plt.cm.Reds]
    
    for policy_name in ['PINN_Policy', 'Poly_Policy', 'Simple_Policy']:
        for i in range(3):
            traj = read_trajectory("Data/Trajectories/" + policy_name + "_trajectory_" + str(i) + ".data")
            if i == 0:
                shape = Eros().model_potatok
                new_fig = True
            else:
                shape = None
                new_fig = False

            op.plot3d(traj[0], traj[1], None, show=False, obj_file=shape, save=False, traj_cm=colors[i], new_fig=new_fig)
        
        # plt.legend()
        plt.gca().view_init(45, 45)
        plt.tight_layout()
        os.makedirs('Plots/', exist_ok=True)
        plt.savefig('Plots/' + policy_name + '_protected_Orbits.pdf')
    plt.show()

if __name__ == "__main__":
    main()