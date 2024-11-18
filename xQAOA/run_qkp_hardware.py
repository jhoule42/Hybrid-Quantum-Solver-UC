#%%
import numpy as np

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from xQAOA.kp_utils import *
from xQAOA.qkp_solver import *
from UC.scripts.utils.visualize import plot_custom_histogram

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_ibm_runtime import SamplerV2

from qiskit_ibm_runtime import QiskitRuntimeService, ibm_backend

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)


#%% PARAMETERS
n = 4
k_range = [15]  # Simplified from np.arange(10, 24, 1)
theta_range = [0]  # Simplified from [0, -0.5, -1]
N_beta, N_gamma = 5, 5  # Number of grid points for beta and gamma


# Initialize a nested dictionary to store results for different methods
results = {
    'very_greedy': {},
    'lazy_greedy': {},
    'hourglass': {},
    'copula': {},
    'X': {}
}

# List of distribution functions
list_distributions = [
    generate_profit_spanner,
    # generate_profit
]

# backend = FakeSherbrooke()

sampler = SamplerV2(backend)
pm = generate_preset_pass_manager(optimization_level=2, backend=backend)


#%%
# Iterate over different distributions
for dist_func in list_distributions:
    print(f"\nUsing distribution: {dist_func.__name__}")

    # Generate values and weights for the current distribution
    v, w = dist_func(n)
    c = np.ceil(np.random.uniform(0.25, 0.75) * sum(w)).astype(int)

    # Solve with Brute Force for optimal solution
    solutions = bruteforce_knapsack(v, w, c)
    bitstrings_ranked = [i[2] for i in solutions]
    optimal_value = solutions[0][0]
    print(f"\nOptimal Solution (BruteForce): {optimal_value}")


    # Run Lazy Greedy Knapsack
    value_LG, weight_LG, bitstring_LG = lazy_greedy_knapsack(v, w, c)
    results['lazy_greedy'][dist_func.__name__] = {
            'ratio_optim': value_LG / optimal_value,
            'rank_solution': bitstrings_ranked.index(bitstring_LG) + 1}
    print(f"Lazy Greedy value: {value_LG}")


    # Run Very Greedy Knapsack
    value_VG, weight_VG, bitstring_VG = very_greedy_knapsack(v, w, c)
    results['very_greedy'][dist_func.__name__] = {
            'ratio_optim': value_VG / optimal_value,
            'rank_solution': bitstrings_ranked.index(bitstring_VG) + 1}
    print(f"Very Greedy value: {value_VG}")


    # X (Standard) MIXER
    print("\nX MIXER") 
    optimizer_X = QKPOptimizer(v, w, c, mixer='X', run_hardware=True,
                               backend=backend, sampler=sampler, pass_manager=pm
                               )
    
    optimizer_X.parameter_optimization(k_range, [0], N_beta, N_gamma)

    # Store results in dict
    results['X'][dist_func.__name__] = {
            'ratio_optim': optimizer_X.best_value / optimal_value,
            'rank_solution': bitstrings_ranked.index(optimizer_X.best_bitstring) + 1,
            'beta_opt': optimizer_X.best_params[0],
            'gamma_opt': optimizer_X.best_params[1]}
    

#     # HOURGLASS MIXER
#     print("\nHOURGLASS MIXER")
#     optimizer_H = QKPOptimizer(v, w, c, mixer='hourglass')
#     optimizer_H.parameter_optimization(k_range, [0], N_beta, N_gamma)

#     # Store results in dict
#     results['hourglass'][dist_func.__name__] = {
#             'ratio_optim': optimizer_H.best_value / optimal_value,
#             'rank_solution': bitstrings_ranked.index(optimizer_H.best_bitstring) + 1,
#             'beta_opt': optimizer_H.best_params[0],
#             'gamma_opt': optimizer_H.best_params[1]}


#     # COPULA MIXER
#     print("\nCOPULA MIXER")
#     optimizer_C = QKPOptimizer(v, w, c, mixer='copula')
#     optimizer_C.parameter_optimization(k_range, [0], N_beta, N_gamma)

#     # Store results in dict
#     results['copula'][dist_func.__name__] = {
#             'ratio_optim': optimizer_C.best_value / optimal_value,
#             'rank_solution': bitstrings_ranked.index(optimizer_C.best_bitstring) + 1,
#             'beta_opt': optimizer_C.best_params[0],
#             'gamma_opt': optimizer_C.best_params[1]}

#%%
# Plot the results
plot_rank_and_ratio(results)


#%%
# Example of running a single instance with specific parameters
bitstring, value, weights, counts = optimizer_X.QKP(
    gamma=0.9424777960769379, 
    beta=0.6283185307179586,
    k=15,
    theta=0)

print(f"Bitstring: {bitstring}")
print(f"Value: {value}")
print(f"Weights: {weights}")
plot_custom_histogram(counts, max_bitstrings=30)
# %%
