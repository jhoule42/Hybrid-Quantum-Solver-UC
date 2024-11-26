#%% Run the algorithm to solve the UC problem using Knapsack

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from UC.scripts.utils.utils import *
from UC.scripts.utils.visualize import *
from UC.scripts.solvers.classical_solver_UC import *

from xQAOA.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *

#%%
L = 100 # Total power demand
n_units = 6  # Number of units

A, B, C, p_min, p_max = generate_units(N=n_units)
param_exec = {"L": L, "n_units":n_units,
              "A": A, "B": B, "C": C,
              "p_min": p_min, "p_max": p_max}

L = np.sum(p_max) * 0.5 # make power load very high


#%%
list_cost = []
list_power = []
list_z_i = []
list_p_i = []

results = {
    'very_greedy': {},
    'lazy_greedy': {},
    # 'hourglass': {},
    'copula': {},
    'X': {}
}

def find_min_D(range_D, A, B, C, L):
    """
    Finds the smallest value of D from range_D such that capacity >= 0.
    
    Parameters:
    range_D: list or np.array of possible D values
    B: constant B in the equation
    C: constant C in the equation
    L: threshold value for capacity
    
    Returns:
    optimal_D: The smallest D that satisfies the condition or None if not found
    """
    for D in range_D:
        # Calculate p_i for current D
        p_i = (D - B) / (2 * C)
        min_val = np.min(A + B*p_i + C*(p_i**2))

        # Compute capacity
        capacity = np.sum(p_i) - L
        
        # Check if capacity is non-negative
        if capacity >= min_val:
            print(f"Found D: {D}, Capacity: {capacity}")
            return D
    
    # If no valid D found
    print("No suitable D found that results in capacity >= 0.")
    return None

min_D = find_min_D(np.linspace(0, 10, 1000), A, B, C, L)
range_D = np.linspace(1, 10, 100)



#%%
for idx, D in enumerate(range_D):
    # print("IDX", idx)
    # print(f"D: {D}")
    # set P_i subject to derivative
    # need to define D such that we don't have negative capacity
    p_i = (D - B) / (2*C)

    # Make sure that the values are within the optimal range
    p_i = np.clip(p_i, p_min, p_max)

    # Knapsack Mapping
    # ∑ v_i * y_i ==> ∑ A*y_i + B*p_i*y_i + C*(p_i*^2)*y_i
    # ∑ w_i * y_i ≤ c  ==>  ∑ p_i * z_i ≤ ∑ p_i - L
    # capacity = np.abs(np.sum(p_i) - L)
    capacity = np.sum(p_i) - L
    # print('Capacity', capacity)
    w = p_i
    v = A + B*p_i + C*(p_i**2)

    # Solve with Brute Force for optimal solution
    solutions = bruteforce_knapsack(v, w, capacity, bit_mapping='inverse')
    bitstrings_ranked = [i[2] for i in solutions]
    optimal_value = solutions[0][0]
    # print(f"\nOptimal Solution (BruteForce): {optimal_value}")
    # print(f"Optimal bitstring: {bitstrings_ranked[0]}")

    # Store results in dict
    results['bruteforce'] = {
            'bitstring': bitstrings_ranked[0],
            # 'ratio_optim': optimal_value / optimal_value,
            'rank_solution': bitstrings_ranked.index(bitstrings_ranked[0]) + 1}

    # X (Standard) MIXER
    print("\nX MIXER")
    optimizer_X = QKPOptimizer(v, w, capacity, mixer='X')
    optimizer_X.parameter_optimization([15], [0], N_beta=10, N_gamma=10, bit_mapping='inverse')

    # Store results in dict
    results['X'] = {
            'bitstring': optimizer_X.best_bitstring,
            'ratio_optim': optimizer_X.best_value / optimal_value,
            'rank_solution': bitstrings_ranked.index(optimizer_X.best_bitstring) + 1,
            'beta_opt': optimizer_X.best_params[0],
            'gamma_opt': optimizer_X.best_params[1]}
    

    # Comput Costs
    z_i = bitstrings_ranked[0]
    z_i = [int(char) for char in z_i]

    cost = np.sum(A*z_i + B*p_i*z_i + C*(p_i**2)*z_i)
    power = np.sum(p_i*z_i) # total power delivered

    list_cost.append(cost)
    list_power.append(power)
    list_p_i.append(p_i*z_i)
    list_z_i.append(z_i)

print("Done.")

# %% PLOT COST VS D
plt.figure()
plt.plot(range_D, list_cost)
plt.xlabel('range D')
plt.ylabel('cost')
plt.show()

#%%
def extract_quantum_results(list_cost, list_power, list_zi, list_p_i, range_D):
    array_cost = np.array(list_cost)
    
    # Find the smallest non-zero cost
    min_array_cost = np.min(array_cost[array_cost != 0])
    
    # Get the index of the smallest non-zero cost
    min_index = np.where(array_cost == min_array_cost)[0][0]
    
    # Create a dictionary to store the results, similar to results_gurobi
    results_quantum = {
        'bitstring': ''.join(map(str, list_zi[min_index])),
        'power': list_p_i[min_index],
        'cost': min_array_cost,
        'optimal_D': range_D[min_index]
    }
    
    return results_quantum

# Call the function and print the results
results_quantum = extract_quantum_results(list_cost, list_power, list_z_i, list_p_i, range_D)
print('Quantum Solver')
print(f"Cost: {results_quantum['cost']:.1f}")
print(f"Power Load: {np.sum(results_quantum['power']):.1f}/{L}")
print(f"Bitstring: {results_quantum['bitstring']}")
print('Power:', results_quantum['power'])
print(f"Optimal D: {results_quantum['optimal_D']}")


# %%
# Compare with classical solution
results_gurobi = gurobi_solver(A, B, C, L, p_min, p_max)
print("Classical Solver")
print(f"Cost: {results_gurobi['cost']:.1f}")
print(f"Power Load: {np.sum(results_gurobi['power']):.1f}/{L}")
# print(f"p_i: {results_gurobi['power']}")
print("Bitstring:", results_gurobi['bitstring'])



#%% Show Power distribution
def visualize_optimal_power_distribution(param_exec, show_details_cost=True, **power_distributions):
    """
    Plots the optimal power distributions from the provided dictionaries.
    Optionally shows unit parameters A, B, C for each unit under the bars.

    Args:
        param_exec (dict): Contains problem parameters like 'p_min', 'p_max', 'A', 'B', 'C'.
        show_details_cost (bool): If True, prints detailed cost calculations.
        **power_distributions: Arbitrary number of power distribution dictionaries with labels as keys.
    """
    
    # Ensure p_min and p_max are numpy arrays for consistent indexing
    p_min = np.array(param_exec['p_min'])
    p_max = np.array(param_exec['p_max'])
    A = param_exec['A']
    B = param_exec['B']
    C = param_exec['C']

    num_units = len(p_min)  # Number of power units
    unit_indices = np.arange(num_units)  # Indices for the units

    # Create the bar plot for the range from p_min to p_max
    plt.figure(figsize=(10, 6))

    # Create bars for the range from p_min to p_max
    bars = plt.bar(unit_indices, p_max - p_min, bottom=p_min, color='lightgrey',
                   edgecolor='black', linewidth=1.2, label='Power Range')

    # Initialize total cost storage for each power distribution
    total_costs = {label: 0 for label in power_distributions}

    # Plot the power distributions
    colors = ['royalblue', 'crimson', 'green', 'purple', 'orange']  # Add more colors if needed
    for idx, (label, power_array) in enumerate(power_distributions.items()):
        power_array = np.array(power_array['power'])  # Ensure it's a numpy array
        print("power array", power_array)
        units_on = (power_array > 0).astype(int)  # 1 if on, 0 if off

        # Plot the optimal power output lines
        for i in range(num_units):
            plt.hlines(y=power_array[i], xmin=bars[i].get_x(),
                       xmax=bars[i].get_x() + bars[i].get_width(),
                       color=colors[idx % len(colors)], linestyle='-', linewidth=3,
                       label=label if i == 0 else "")

        # Calculate and print detailed costs (if enabled)
        if show_details_cost:
            print(f'\n{label}')
            for i in range(num_units):
                cost = A[i] * units_on[i] + B[i] * power_array[i] + C[i] * (power_array[i] ** 2)
                total_costs[label] += cost

                # Format the output for better alignment
                print(f"Unit {i+1}: {A[i]:>6.1f} x {units_on[i]} + "
                      f"{B[i]:>6.2f} x {power_array[i]:>8.2f} + "
                      f"{C[i]:>8.3f} x {power_array[i]**2:>8.2f} = {cost:>30.2f}")

            print(f"{'Total ' + label + ' Cost:':>43} {total_costs[label]:.2f}")


    # Customize the plot
    plt.xlabel('Power Units', fontsize=14)
    plt.ylabel('Power Output', fontsize=14)
    plt.title('Optimal Power Distribution', fontsize=16, fontweight='bold')
    plt.ylim(bottom=0)
    plt.xticks(unit_indices, [f'Unit {i+1}' for i in unit_indices], fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


visualize_optimal_power_distribution(param_exec,
                                     Gurobi = results_gurobi,
                                     Quantum = results_quantum
                                     )


# %%s
