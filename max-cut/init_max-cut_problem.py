# Define the Max-Cut problem to solve

#%%
import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph

from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz



#%% ---------------- Create Graph ----------------

n = 5 # nodes number
graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n, 1))
edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
graph.add_edges_from(edge_list)
draw_graph(graph, node_size=600, with_labels=True)


# %%
# Map classical imputs to a quantum problem
# !!! Regarder plus en detail comment le mapping se fait !!!
def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of 'build_max_cut_graph'.
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list


# Map the graph to a Pauli list for Max Cut
max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
print("Cost Function Hamiltonian:", cost_hamiltonian)

# %% ---------------- Create the QAOA circuit ----------------
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
circuit.decompose().draw('mpl') # see the circuit details


# %% ---------------- Save quantum circuit to disk ----------------
with open('circuits/test.qpy', 'wb') as file:
    qpy.dump(circuit, file)

print('Done.')
# %%