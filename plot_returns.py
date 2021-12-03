import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GravNN.Visualization.VisualizationBase import VisualizationBase
import matplotlib.lines as mlines
from scipy.signal import savgol_filter
def plot_curve(stats, label, color):
    steps = stats['EnvironmentSteps']
    times = stats['Time']
    returns = stats['AverageReturn']
    SE = stats['AverageReturnSE']

    # try:
    #     returns = savgol_filter(returns.values, 9, 3)
    #     SE = savgol_filter(SE.values, 9, 3)
    # except: 
    #     pass


    plt.plot(times, returns, color=color, label=label)
    plt.fill_between(times, 
                    (returns-SE), 
                    (returns+SE), 
                    color=color, alpha=0.5)

def plot_scatter(stats, step_list, marker_list, color):
    steps = stats['EnvironmentSteps']
    times = stats['Time']
    returns = stats['AverageReturn']

    for i in range(len(step_list)):
        index = steps[steps==step_list[i]].index
        plt.scatter(times[index], returns[index], marker=marker_list[i], color=color, zorder=99)

def main():
    stats_pinn = pd.read_pickle("Data/Policies/PINN_Policy/metrics.data")
    stats_poly = pd.read_pickle("Data/Policies/Poly_Policy/metrics.data")
    stats_simple = pd.read_pickle("Data/Policies/Simple_Policy/metrics.data")

    print(stats_pinn[stats_pinn['AverageReturn'] == stats_pinn['AverageReturn'].max()])
    print(stats_poly[stats_poly['AverageReturn'] == stats_poly['AverageReturn'].max()])
    print(stats_simple[stats_simple['AverageReturn'] == stats_simple['AverageReturn'].max()])

    vis = VisualizationBase(formatting_style='AIAA')
    vis.fig_size = vis.AIAA_full_page
    plt.rc('font', size= 10.0)

    vis.newFig()
    plot_curve(stats_pinn, "PINN", "blue")
    plot_curve(stats_poly, "Polyhedral", "red")
    plot_curve(stats_simple, "Simple", "green")
    plt.xlabel("Time [s]")
    plt.ylabel("Average Return")
    first_legend = plt.legend(loc='upper left')
    plt.gca().add_artist(first_legend)


    step_list = [50000, 100000, 200000]
    marker_list = ['o', '^', '*']
    plot_scatter(stats_pinn, step_list, marker_list, 'navy')
    plot_scatter(stats_poly, step_list, marker_list, 'darkred')
    plot_scatter(stats_simple, step_list, marker_list, 'darkgreen')

    legend_list = []
    for i in range(len(step_list)):
        line = mlines.Line2D([], [], color='black', marker=marker_list[i], linestyle='None',
                            markersize=10, label=str(step_list[i]) + ' Steps')
        legend_list.append(line)
    plt.legend(handles=legend_list, loc='upper right')
    vis.save(plt.gcf(), "Returns.pdf")
    plt.show()


if __name__ == '__main__':
    main()