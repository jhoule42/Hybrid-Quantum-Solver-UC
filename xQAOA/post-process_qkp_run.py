#%%
import numpy as np
import json
import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

from qiskit_ibm_runtime import QiskitRuntimeService

from xQAOA.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *
from UC.scripts.utils.visualize import plot_custom_histogram, plot_value_distribution


#%% ===================== Connect to the Backend =====================
service = QiskitRuntimeService(channel="ibm_quantum",
                               instance='pinq-quebec-hub/universit-de-cal/main')
print("Connected to the backend.")


#%% ===================== PARAMETERS =====================

PATH_RUNS = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA/runs"
func_name = "generate_profit_spanner"
exec_time = "2024-11-22_16-33-43"  # execution directory
workload_id = "cx0ha5gpx23g008j8z1g"

# Load Parameters
with open(f'{PATH_RUNS}/{exec_time}/parameters.json', 'r') as file:
    dict_params = json.load(file)
v = np.array(dict_params[func_name]['v'])
w = np.array(dict_params[func_name]['w'])
c = dict_params[func_name]['c']

n_units = len(v)
fake_hardware = dict_params['fake_hardware']
N_beta = dict_params['N_beta']
N_gamma = dict_params['N_gamma']
bit_mapping = dict_params['bit_mapping']
k_range = dict_params['k_range']
theta_range = dict_params['theta_range']


# Load Jobs ID
with open(f'{PATH_RUNS}/{exec_time}/dict_jobs.json', 'r') as file:
    dict_jobs = json.load(file)
print("Jobs Loaded.")


# Initialize a nested dictionary to store results for different methods
results = {}
results = {'very_greedy': {},
           'very_greedy': {},
           'lazy_greedy': {},
           'X': {},
           'copula': {},
           }


#%% ===================== CLASSICAL SOLUTIONS =====================
# Solve with Brute Force for optimal solution
bruteforce_bitstring = bruteforce_knapsack(v, w, c)
bitstrings_ranked = [i[2] for i in bruteforce_bitstring]
bruteforce_value = bruteforce_bitstring[0][0]
print(f"\nOptimal Solution (BruteForce): {bruteforce_value}")

# Compute Lazy Greedy Knapsack
value_LG, weight_LG, bitstring_LG = lazy_greedy_knapsack(v, w, c)
results['lazy_greedy'][func_name] = {
        'ratio_optim': value_LG / bruteforce_value,
        'rank_solution': bitstrings_ranked.index(bitstring_LG) + 1}
print(f"Lazy Greedy value: {value_LG}")

# Compute Very Greedy Knapsack
value_VG, weight_VG, bitstring_VG = very_greedy_knapsack(v, w, c)
results['very_greedy'][func_name] = {
        'ratio_optim': value_VG / bruteforce_value,
        'rank_solution': bitstrings_ranked.index(bitstring_VG) + 1}
print(f"Very Greedy value: {value_VG}")


#%% ===================== Post-Process Hardware JOBS =====================

# Extract jobs
if not fake_hardware:
    print("Extracting jobs.")
    jobs = service.jobs(limit=400,
                        backend_name='ibm_quebec',
                        session_id='cx0ha5gpx23g008j8z1g')
else:
    jobs = dict_jobs

best_value = -np.inf

# get access to all the job results
dict_results = {}
for job in jobs:

    if not fake_hardware:
        job_id = job.job_id()
        params = dict_jobs.get(job_id)
        result = job.result()[0]
        counts = result.data.meas.get_counts()

    else:
        job_id = job
        counts = dict_jobs[job_id]['counts']
        params = dict_jobs[job_id]['params']

    # Find best solution
    bitstring = max(counts, key=counts.get)

    if bit_mapping == 'regular':
        value = sum(int(bitstring[i]) * v[i] for i in range(n_units))
        weight = sum(int(bitstring[i]) * w[i] for i in range(n_units))
    
    if bit_mapping == 'inverse':
        value = sum((1 - int(bitstring[i])) * v[i] for i in range(n_units))
        weight = sum((1 - int(bitstring[i])) * w[i] for i in range(n_units))

    # Make sure the solution is valid
    if weight <= c:
        if value == best_value:
            list_best_bitstring.append(bitstring)
            list_best_weight.append(weight)
            list_best_params.append(params)
            list_job_id.append(job_id)
        
        if value > best_value: # update if we find a better solution
            list_best_bitstring = [bitstring]
            list_best_weight = [weight]
            list_best_params = [params]
            list_job_id = [job_id]

            best_value = value
            print('New best value', best_value)
            best_bitstring = bitstring
            best_weight = weight
            best_params = params

print("\nxQAOA Solution Quantum")
print(f"Best Value: {best_value}")
print(f"Bitstring: {best_bitstring}")
print(f"Weight: {weight}")
print(f"Nb of optimal solution: {len(list_best_bitstring)}")

#%%
# Store results in dict
results['copula'][func_name] = {
        'ratio_optim': best_value / bruteforce_value,
        'rank_solution': bitstrings_ranked.index(best_bitstring) + 1,
        'beta_opt': best_params['params'][0],
        'gamma_opt': best_params['params'][1],
        'best_value': best_value,
        }

# Plot the results
plot_rank_and_ratio(results, methods=['lazy_greedy', 'very_greedy', 'copula'],
                    labels = ['LG', 'VG', r'$QKP_{COP}$'])

#%%
# Visualize distribution of best parameters
for params in list_best_params:

    if fake_hardware:
        # access counts results that are saved to file
        job_id = list_job_id[0]
        bitstring = list_best_bitstring[0]
        counts = dict_jobs[job_id]['counts']
        value = results['copula'][func_name]['best_value']

    plot_custom_histogram(counts, max_bitstrings=40,
                            remove_xticks=True, display_text=False)

    combined_data = []
    for value, weight, bitstring in bruteforce_bitstring:
        # Get the count from counts dictionary; use 0 if the bitstring is not found
        bitstring_count = counts.get(bitstring, 0)
        combined_data.append((value, bitstring_count, bitstring))

    # Usefull when we try to scale and we are limited by shots number
    plot_value_distribution(combined_data,
                            optimal_value=bruteforce_value,
                            best_val_found=results['copula']['generate_profit_spanner']['best_value'])
# %%
