#%% Reproduction de la figrue 5 du paper
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from xQAOA.kp_utils import *
from xQAOA.scripts.experiments.solvers.qkp_solver import *
from UC.scripts.utils.visualize import plot_custom_histogram

#%% plot figure 5 in the paper (Smooth Lazy Greedy)
# Define the logistic function for fitting
def logistic_fit(r, C, k, r_star):
    return 1 / (1 + C * np.exp(-k * (r - r_star)))

# Knapsack data
v = [3627, 580, 1835, 246, 364, 674, 840, 1391, 250, 193]
w = [6012, 1297, 2148, 642, 678, 895, 1012, 1365, 502, 452]
c = 10240
r = np.array(v) / np.array(w)

# Values of k to plot
k_values = [0, 5, 10, 20]


# Plotting
plt.figure(figsize=(10, 8))
for k in k_values:
    # Calculate probabilities using the logistic bias function
    p_i = logistic_bias(v, w, c, k)
    
    # Plot original data points
    plt.scatter(r, p_i, label=f'k = {k}')

    # Fit a curve to the data points
    popt, _ = curve_fit(lambda r, C, r_star: logistic_fit(r, C, k, r_star), r, p_i)
    C_fit, r_star_fit = popt

    # Generate smooth curve for the fit
    r_smooth = np.linspace(0, max(r), 100)
    p_smooth = logistic_fit(r_smooth, C_fit, k, r_star_fit)
    plt.plot(r_smooth, p_smooth, linestyle='--')

plt.xlabel(r'$r_i = v_i / w_i$', fontsize=12)
plt.ylabel(r'$p_i$', fontsize=12)
plt.title('Smoothed Lazy Greedy Distribution using Logistic Function')
plt.grid(True)
plt.legend()
plt.show()
# %%
