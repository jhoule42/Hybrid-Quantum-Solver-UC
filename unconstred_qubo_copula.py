#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram, plot_distribution
import pennylane as qml
import ast
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")
from xQAOA.kp_utils import *
from xQAOA.scripts.solvers.qkp_solver import *


#%% ================ LOAD HAMILTONIAN FROM FILE ================

PATH_TO_SAVE = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA"
file_path = "hamiltonian_file.json"

# Open and read the JSON file
with open(file_path, "r") as file:
    data = json.load(file)

# Access the data
values = data["items_values"]
weights = data["items_weights"]
capacity = int(data["capacity"])
h_terms = data["h_terms"]
j_terms = data["j_terms"]

# Convert keys from strings to tuples
h_terms = {ast.literal_eval(k): v for k, v in h_terms.items()}
j_terms = {ast.literal_eval(k): v for k, v in j_terms.items()}


def create_sparse_pauli_op(h_terms, j_terms):
    paulis = []
    coeffs = []

    # Handle linear terms (h_terms)
    for (i,), h_term in h_terms.items():
        pauli_str = 'I' * len(h_terms)
        pauli_str = pauli_str[:i] + 'Z' + pauli_str[i+1:]
        paulis.append(pauli_str)
        coeffs.append(h_term)

    # Handle quadratic terms (j_terms)
    for (i, j), j_term in j_terms.items():
        pauli_str = 'I' * len(h_terms)
        pauli_str = list(pauli_str)
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'
        paulis.append(''.join(pauli_str))
        coeffs.append(j_term)

    return SparsePauliOp(paulis, coeffs)

# Create the SparsePauliOp
hamiltonian = create_sparse_pauli_op(h_terms, j_terms)

#%% ====================== BRUTEFORCE SOLUTION ======================

bruteforce_solutions = bruteforce_knapsack(values, weights, capacity)
opt_val, opt_weight, opt_bitstring = ([i[0] for i in bruteforce_solutions][0],
                                      [i[1] for i in bruteforce_solutions][0],
                                      [i[2] for i in bruteforce_solutions][0])
print(f"Brute Force Solution: {opt_val} | {opt_weight} | {opt_bitstring}")


#%% ======================  COPULA MIXER  ======================

# Parameters
shots = 5000  # Number of samples used
k_range = [15]
theta_range = [1]
p = 30
betas = np.linspace(0, 1, p)[::-1]  # Parameters for the mixer Hamiltonian
gammas = np.linspace(0, 1, p)  # Parameters for the cost Hamiltonian (Our Knapsack problem)
x0_params = np.empty(2*p, dtype=float)
x0_params[0::2] = betas # gammas
x0_params[1::2] = gammas # betas

# COPULA MIXER
print("\nCOPULA MIXER")
optimizer_C = QKPOptimizer(values, weights, capacity,
                           mixer='X',
                           optimal_solution=opt_val,
                           speedup_computation=False,
                           p=p,
                           h_term=h_terms, J_term=j_terms)


# TODO Comparer avec ce que j'obtenais avec la fonction QKP
result = optimizer_C.QKP(betas, gammas, k=15, theta=-1, bit_mapping='regular',
                      cost_hamiltonian=True)

#%%
# Extract Restuls
bitstring = result[0]
vals = int(result[1])
weight = result[2]
counts = result[3]
valid_solution = result[4]
circuit = result[5]

#%%
plot_custom_histogram(counts, max_bitstrings=1000, display_text=False, remove_xticks=True)

values_unbalanced_qiskit = {sum_values(sample_i, values): count
                    for sample_i, count in counts.items()
                    if sum_weight(sample_i, weights) <= capacity
                    }

plot_histogram_with_vlines(values_unbalanced_qiskit,
                           -opt_val,
                           log=False,
                           bins_width=1000)



#%%

def cost_func_qaoa(params, ansatz, hamiltonian, estimator):
    """Evaluate the cost function using the estimator to run QAOA."""

    pub = (ansatz, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs
    return cost # Return the first (and only) expectation value


# The circuit needs to be transpiled to the AerSimulator target
pass_manager = generate_preset_pass_manager(3, AerSimulator())
isa_circuit = pass_manager.run(qc) # instruction set architecture circuit

init_params = [float(i) for i in x0_params]

exact_estimator = Estimator()
result = minimize(cost_func_qaoa,
                  x0 = init_params,
                  args=(isa_circuit, hamiltonian, exact_estimator),
                  method="COBYLA",
                  options={'maxiter': 2000, 'disp': False})
print(result)

# Sampling
exact_sampler = Sampler()
optimized_circuit = isa_circuit.assign_parameters(result.x)
optimized_circuit.draw('mpl', fold=False, idle_wires=False)

measured_circuit = isa_circuit.copy()
measured_circuit.measure_all() # adding measurments

# The circuit needs to be transpiled to the AerSimulator target
pass_manager = generate_preset_pass_manager(3, AerSimulator())
isa_circuit = pass_manager.run(measured_circuit)
pub = (isa_circuit, result.x, 5000)
job = exact_sampler.run([pub])
job_result = job.result()
pub_result = job_result[0]

counts = pub_result.data.meas.get_counts()


#%%
plot_custom_histogram(counts, max_bitstrings=1000, display_text=False, remove_xticks=True)

values_unbalanced_qiskit = {sum_values(sample_i, values): count
                    for sample_i, count in counts.items()
                    if sum_weight(sample_i, weights) <= capacity
                    }
# print(f"The number of solutions using unbalanced penalization is {counts.get(opt_bitstring)} out of {shots}")

plot_histogram_with_vlines(values_unbalanced_qiskit,
                           -opt_val,
                           log=False,
                           bins_width=200)



#%% ======================  X MIXER  ======================

shots = 5000  # Number of samples used
k_range = [15]
theta_range = [-1]
p = 3
betas = np.linspace(0, 1, p)[::-1]  # Parameters for the mixer Hamiltonian
gammas = np.linspace(0, 1, p)  # Parameters for the cost Hamiltonian (Our Knapsack problem)

# COPULA MIXER
print("\nCOPULA MIXER")
optimizer_C = QKPOptimizer(values, weights, capacity,
                           mixer='X',
                           optimal_solution=opt_val,
                           speedup_computation=False,
                           p=p,
                           h_term=h_terms, J_term=j_terms)

best_solution, best_value, total_weight, counts, sucess, qc =optimizer_C.QKP(betas, gammas, k=15, theta=-1, bit_mapping='regular',
                cost_hamiltonian=True)

plot_custom_histogram(counts, max_bitstrings=1000, display_text=False, remove_xticks=True)

values_unbalanced_qiskit = {sum_values(sample_i, values): count
                    for sample_i, count in counts.items()
                    if sum_weight(sample_i, weights) <= capacity
                    }
# print(f"The number of solutions using unbalanced penalization is {counts.get(opt_bitstring)} out of {shots}")

plot_histogram_with_vlines(values_unbalanced_qiskit,-opt_val,
                           log=False,
                           bins_width=1000)





# %%
