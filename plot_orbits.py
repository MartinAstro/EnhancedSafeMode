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
    
    pinn_traj = read_trajectory("Data/Policies/PINN_Policy/trajectory.data")
    poly_traj = read_trajectory("Data/Policies/Poly_Policy/trajectory.data")
    simple_traj = read_trajectory("Data/Policies/Simple_Policy/trajectory.data")
    
    colors = [plt.cm.winter, plt.cm.autumn, plt.cm.summer]
    colors = [plt.cm.winter, plt.cm.cool, plt.cm.Wistia]
    colors = [plt.cm.winter, plt.cm.summer, plt.cm.autumn]

    op.plot3d(pinn_traj[0], pinn_traj[1], None, show=False, obj_file=Eros().model_potatok, save=False, traj_cm=colors[0], reverse_cmap=True)
    op.plot3d(poly_traj[0], poly_traj[1], None, show=False, save=False, traj_cm=colors[1], new_fig=False, reverse_cmap=True)
    op.plot3d(simple_traj[0], simple_traj[1], None, show=False, save=False, traj_cm=colors[2], new_fig=False, reverse_cmap=True)


    # Creating legend with color box
    x_space = np.linspace(0, 1, 100)
    custom_lines = [mlines.Line2D(x_space, [0], color=colors[0](1), lw=2, label="PINN"),
                    mlines.Line2D(x_space, [0], color=colors[1](1), lw=2, label="Polyhedral"),
                    mlines.Line2D(x_space, [0], color=colors[2](1), lw=2, label="Simple")]

    plt.legend(handles=custom_lines)
    plt.gca().view_init(15, 15)
    plt.tight_layout()
    os.makedirs('Plots/', exist_ok=True)
    plt.savefig('Plots/Agent_Orbits.pdf')
    plt.show()

if __name__ == "__main__":
    main()