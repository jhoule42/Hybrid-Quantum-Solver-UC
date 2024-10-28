# QAOAO Algorithm Simulation for the Knapsack Problem.
# Author: Julien-Pierre Houle
#%%

from itertools import product
import matplotlib.pyplot as plt
import networkx as nx  # noqa
import numpy as np
import pandas as pd


#%%
# Create the problems graph

# # Classiq network version for the electric grid problem
# # building data matrix, it doesn't need to be a symmetric matrix.
# cost_matrix = np.array([[0.5, 1.0, 1.0, 2.1], [1.0, 0.6, 1.4, 1.0], [1.0, 1.4, 0.4, 2.3]])

# Sources = ["A1", "A2", "A3"]
# Consumers = ["B1", "B2", "B3", "B4"]

# N = len(Sources) # number of sources
# M = len(Consumers) # number of consumers

# graph = nx.DiGraph()
# graph.add_nodes_from(Sources + Consumers)
# for n, m in product(range(N), range(M)):
#     graph.add_edges_from([(Sources[n], Consumers[m])], weight=cost_matrix[n, m])


# # Plot the graph
# plt.figure(figsize=(10, 6))
# left = nx.bipartite.sets(graph)[0]
# pos = nx.bipartite_layout(graph, left)

# nx.draw_networkx(graph, pos=pos, nodelist=Consumers, font_size=22, font_color="None")
# nx.draw_networkx_nodes(graph, pos, nodelist=Consumers, node_color="#119DA4", node_size=500)

# for fa in Sources:
#     x, y = pos[fa]
#     plt.text(x, y, s=fa, bbox=dict(facecolor="#F43764", alpha=1),
#              horizontalalignment="center", fontsize=15)

# nx.draw_networkx_edges(graph, pos, width=2)
# labels = nx.get_edge_attributes(graph, "weight")
# nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=12)
# nx.draw_networkx_labels(graph, pos, labels={co: co for co in Consumers}, font_size=15, font_color="#F4F9E9")

# plt.axis("off")
# plt.show()

import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz


# n = 5

# graph = rx.PyGraph()
# graph.add_nodes_from(np.arange(0, n, 1))
# edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
# graph.add_edges_from(edge_list)
# draw_graph(graph, node_size=600, with_labels=True)


# %%
# 1) Map the classical output to the quantum problem

# Hamiltonian Cost ?

from qiskit.quantum_info import SparsePauliOp
# def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
#     """Convert the graph to Pauli list.

#     This function does the inverse of `build_max_cut_graph`
#     """
#     pauli_list = []
#     for edge in list(graph.edge_list()):
#         paulis = ["I"] * len(graph)
#         paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

#         weight = graph.get_edge_data(edge[0], edge[1])

#         pauli_list.append(("".join(paulis)[::-1], weight))

#     return pauli_list

# max_cut_paulis = build_max_cut_paulis(graph)


# !!! Changer cette fonction pour Knapsack Problem !!!
cost_hamiltonian = SparsePauliOp.from_list([("II", 2), ("XX", -3), ("YY", 2), ("ZZ", -4)])
print("Cost Function Hamiltonian:", cost_hamiltonian)



#%%
# 2) Convert Hamiltonian to Quantum Circuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit import transpile

# circuit with depth p=1 (one layer)
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
circuit.measure_all()
circuit.draw('mpl')
# print(circuit.parameters)


#%%
# 3) Run Simulation
from qiskit_aer import AerSimulator

simulator = AerSimulator()


qc = QuantumCircuit(2,1)
qc.h(1)
qc.measure(1,0)
qc.h(0).c_if(0,1)
qc.save_statevector()
qc.draw()

qc_aer = transpile(qc, backend=simulator)

results = simulator.run(qc_aer).result()
counts = results.get_counts()
print(counts)


#%%
# Adding noise to the circuit
from qiskit.visualization import plot_distribution

result_ideal = simulator.run(qc_aer, shots = 1024).result()
counts_ideal = result_ideal.get_counts()
plot_distribution(counts_ideal)


#%%
from qiskit_aer import noise
prob = 0.1
error = noise.depolarizing_error(prob, 2)

noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error, ['cx'])
basis_gates = noise_model.basis_gates

result_noisy = simulator.run(qc,
                            shots = 1024,
                            noise_model=noise_model,
                            basis_gates=basis_gates).result()

counts_noisy = result_noisy.get_counts()
plot_distribution([counts_ideal, counts_noisy], title='Noiseless vs. Noisy Simulated counts')








# %%
# Hardware optimisation (transpilation)
# !!! Revenir plus tart sur ce point !!!

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# QiskitRuntimeService.save_account(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>", overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(min_num_qubits=127)
print(backend)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

candidate_circuit = pm.run(circuit)
candidate_circuit.draw('mpl', fold=False, idle_wires=False)


#%%

def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)


    return cost