import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GravNN.Visualization.VisualizationBase import VisualizationBase
import matplotlib.lines as mlines

def plot_curve(stats, label, color):
    steps = stats['EnvironmentSteps']
    times = stats['Time']
    returns = stats['AverageReturn']
    SE = stats['AverageReturnSE']
    plt.plot(times, returns, color=color, label=label)
    plt.fill_between(times, 
                    (returns-SE), 
                    (returns+SE), 
                    color=color, alpha=0.5)

def plot_scatter(stats, step_list, marker_list):
    steps = stats['EnvironmentSteps']
    times = stats['Time']
    returns = stats['AverageReturn']

    for i in range(len(step_list)):
        index = steps[steps==step_list[i]].index
        plt.scatter(times[index], returns[index], marker=marker_list[i], color='black', zorder=99)

def main():
    stats_pinn = pd.read_pickle("Data/Policies/PINN_Policy/metrics.data")
    stats_poly = pd.read_pickle("Data/Policies/Poly_Policy/metrics.data")
    stats_simple = pd.read_pickle("Data/Policies/Simple_Policy/metrics.data")

    vis = VisualizationBase()
    vis.newFig()
    plot_curve(stats_pinn, "PINN", "blue")
    plot_curve(stats_poly, "Polyhedral", "red")
    plot_curve(stats_simple, "Simple", "green")
    plt.xlabel("Time [s]")
    plt.ylabel("Average Return")
    first_legend = plt.legend(loc='upper left')
    plt.gca().add_artist(first_legend)


    step_list = [50, 100, 150]
    marker_list = ['o', '^', '*']
    plot_scatter(stats_pinn, step_list, marker_list)
    plot_scatter(stats_poly, step_list, marker_list)
    plot_scatter(stats_simple, step_list, marker_list)

    legend_list = []
    for i in range(len(step_list)):
        line = mlines.Line2D([], [], color='black', marker=marker_list[i], linestyle='None',
                            markersize=10, label=str(step_list[i]) + ' Steps')
        legend_list.append(line)
    plt.legend(handles=legend_list)
    plt.show()


if __name__ == '__main__':
    main()