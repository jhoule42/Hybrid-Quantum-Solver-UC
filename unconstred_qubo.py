#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram, plot_distribution
import pennylane as qml
import ast

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")
from xQAOA.kp_utils import *


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



#%% ====================== BRUTEFORCE SOLUTION ======================

bruteforce_solutions = bruteforce_knapsack(values, weights, capacity)
opt_val, opt_weight, opt_bitstring = ([i[0] for i in bruteforce_solutions][0],
                                      [i[1] for i in bruteforce_solutions][0],
                                      [i[2] for i in bruteforce_solutions][0])
print(f"Brute Force Solution: {opt_val} | {opt_weight} | {opt_bitstring}")

#%% ====================== CONSTRUCT QUBO ======================

Q = -np.diag(values)  # Matrix Q for the problem above.
x_opt = np.array([[int(bit)] for bit in opt_bitstring])
min_cost = (x_opt.T @ Q @ x_opt)[0, 0]  # using Equation 3 above
print(f"Q={Q}")
print(f"The minimum cost is  {min_cost}")


#%% ====================== CONSTRUCT QUBO 2 - SLACK VARIABLES ======================
N = round(np.ceil(np.log2(capacity)))  # number of slack variables
weights_slack = weights + [2**k for k in range(N)]

QT = np.pad(Q, ((0, N), (0, N)))  # adding the extra slack variables at the end of the Q matrix
n_qubits = len(QT)
lambd = 2  # We choose a lambda parameter enough large for the constraint to always be fulfilled
# Adding the terms for the penalty term
for i in range(len(QT)):
    QT[i, i] += lambd * weights_slack[i] * (weights_slack[i] - 2 * capacity)  # Eq. 10
    for j in range(i + 1, len(QT)):
        QT[i, j] += 2 * lambd * weights_slack[i] * weights_slack[j]  # Eq. 9
offset = lambd * capacity**2
print(f"Q={QT}")

# optimal string slack string
slack_string = np.binary_repr(capacity - opt_weight, N)[::-1]
x_opt_slack = np.concatenate((x_opt, np.array([[int(i)] for i in slack_string]))) # combining the optimal string and slack string
opt_str_slack = "".join(str(i[0]) for i in x_opt_slack)
cost = (x_opt_slack.T @ QT @ x_opt_slack)[0, 0] + offset  # Optimal cost using equation 3
print(f"Cost:{cost}")


#%% ======================  QAOA circuit  ======================

shots = 5000  # Number of samples used
dev = qml.device("default.qubit", shots=shots)
# dev = qml.device("qiskit.aer", wires=5, shots=shots)

@qml.qnode(dev)
def qaoa_circuit(gammas, betas, h, J, num_qubits):
    # Normalize the Hamiltonian
    wmax = max(np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values()))))
    p = len(gammas)

    # Apply the initial layer of Hadamard gates to all qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    # repeat p layers the circuit shown in Fig. 1
    for layer in range(p):
        # ---------- COST HAMILTONIAN ----------
        for ki, v in h.items():  # single-qubit terms
            qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
        for kij, vij in J.items():  # two-qubit terms
            qml.CNOT(wires=[kij[0], kij[1]])
            qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
            qml.CNOT(wires=[kij[0], kij[1]])
        # ---------- MIXER HAMILTONIAN ----------
        for i in range(num_qubits):
            qml.RX(-2 * betas[layer], wires=i)
    return qml.sample()


def qaoa_circuit_qiskit(gammas, betas, h, J, num_qubits):
    """ Create a QAOA circuit using Qiskit."""
    # Normalize the Hamiltonian
    wmax = max(np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values()))))
    p = len(gammas) # nb of qaoa layers
    
    # Create quantum circuit
    qc = QuantumCircuit(num_qubits)
    
    # Initial layer of Hadamard gates
    for i in range(num_qubits):
        qc.h(i)
    # qc.barrier()
    
    # Repeat p layers of QAOA circuit
    for layer in range(p):
        # Cost Hamiltonian
        # Single-qubit terms
        for ki, v in h.items():
            qc.rz(2 * gammas[layer] * v / wmax, qubit=ki[0])
        # qc.barrier()
        # Two-qubit terms
        for kij, vij in J.items():
            qc.cx(kij[0], kij[1])
            qc.rz(2 * gammas[layer] * vij / wmax, kij[1])
            qc.cx(kij[0], kij[1])
        # qc.barrier()
        # Mixer Hamiltonian
        for i in range(num_qubits):
            qc.rx(-2 * betas[layer], qubit=i)
    return qc


def samples_dict(samples, n_items):
    """Just sorting the outputs in a dictionary"""
    results = defaultdict(int)
    for sample in samples:
        results["".join(str(i) for i in sample)[:n_items]] += 1
    return results


#%% ================ RUN QUBO WITH SLACK VARIABLE ================

p = 1
betas = np.linspace(0, 1, p)[::-1]
gammas = np.linspace(0, 1, p)

z_exp = [(1 if i == 0 else -1) for i in x_opt_slack]  # Converting the optimal solution from (0,1) to (1, -1)
h, J, zoffset = from_Q_to_Ising(QT, offset)  # Eq.13 for our problem
energy = energy_Ising(z_exp, h, J, zoffset)  # Caluclating the energy (Should be the same that for the QUBO)
print(f"Minimum energy:{energy}")

# # RUN QAOA PennyLane
# qc_qml = qaoa_circuit(gammas, betas, h, J, num_qubits=len(QT))
# # fig, ax = qml.draw_mpl(qaoa_circuit)(gammas, betas, h, J, len(QT))
# # fig.show()


# samples_slack = samples_dict(qc_qml, n_qubits)
# values_slack = {sum_values(sample_i, values): count
#                 for sample_i, count in samples_slack.items()
#                 # if sum_weight(sample_i, weights_slack) <= capacity
#                 }  # saving only the solutions that fulfill the constraint


# RUN QAOA QISKIT
qc = qaoa_circuit_qiskit(gammas, betas, h, J, num_qubits=len(QT))
qc.measure_active() # Sample quantum circuit
qc.draw('mpl', fold=-1)

sampler = StatevectorSampler()
job = sampler.run([qc], shots=shots)
result = job.result()[0]

counts = result.data.measure.get_counts()
best_solution = max(counts, key=counts.get) # find bitstring with most counts

values_slack2 = {sum_values(sample_i, values): count
                 for sample_i, count in counts.items()
                 if sum_weight(sample_i, weights_slack) <= capacity}

# print(f"The number of optimal solutions using slack variables is {samples_slack[opt_str_slack]} out of {shots}")
print(f"The number of optimal solutions using slack variables is {counts.get(opt_str_slack)} out of {shots}")



#%% ==================== RUN UNBALANCED PENELIZATION ====================

print("Unbalance Penalization Solution")
p = 5
betas = np.linspace(0, 1, p)[::-1]  # Parameters for the mixer Hamiltonian
gammas = np.linspace(0, 1, p)  # Parameters for the cost Hamiltonian (Our Knapsack problem)

fig, ax = plt.subplots()
ax.plot(betas, label=r"$\beta_i$", marker="o", markersize=8, markeredgecolor="black")
ax.plot(gammas, label=r"$\gamma_i$", marker="o", markersize=8, markeredgecolor="black")
ax.set_xlabel("i", fontsize=18)
ax.legend()
fig.show()



#%% ================== PennyLane Implementation ==================

print("PennyLane Implementation")
qc_qml = qaoa_circuit(gammas, betas, h_terms, j_terms, num_qubits=len(values))
# dev._circuit.draw(output="mpl", fold=-1)

# Sampling QML circuit
counts_qml = samples_dict(qc_qml, len(values))

values_unbalanced_qml = {sum_values(sample_i, values): count
                    for sample_i, count in counts_qml.items()
                    if sum_weight(sample_i, weights) <= capacity
                    }
print(f"The number of solutions using unbalanced penalization is {counts_qml[opt_bitstring]} out of {shots}")

# PennyLane Implementation
# plot_histogram_with_vlines(values_unbalanced_qml, values_slack, min_cost)

#%% ================== Qiskit Implementation ==================

print("Qiskit Implementation")
qc = qaoa_circuit_qiskit(gammas, betas, h_terms, j_terms, num_qubits=len(values))

# Sampling Qiskit Circuit
qc.measure_all()
qc.draw('mpl', fold=-1, plot_barriers=False)

sampler = StatevectorSampler()
job = sampler.run([qc], shots=shots)
result = job.result()[0]

counts_qiskit = result.data.meas.get_counts()
counts_qiskit2 = reverse_bits(counts_qiskit)


# plot_custom_histogram(counts_qiskit2, max_bitstrings=1000, display_text=False, remove_xticks=True)

plot_custom_histogram(counts_qiskit2, max_bitstrings=1000, display_text=False, remove_xticks=True)




## %% ========================== VISUALIZE SOLUTION ==========================

best_solution = max(counts_qiskit2, key=counts_qiskit2.get) # find bitstring with most counts

values_unbalanced_qiskit = {sum_values(sample_i, values): count
                    for sample_i, count in counts_qiskit2.items()
                    # if sum_weight(sample_i, weights) <= capacity
                    }
print(f"The number of solutions using unbalanced penalization is {counts_qiskit2.get(opt_bitstring)} out of {shots}")

plot_histogram_with_vlines(values_unbalanced_qiskit,
                           -opt_val,
                           log=False,
                           bins_width=5000)




# %%

