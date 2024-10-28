""" Post-process results for the max-cut problem """
#%%
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
from qiskit_ibm_runtime import RuntimeDecoder


with open("results/result.json", "r") as file:
    data = json.load(file, cls=RuntimeDecoder)

# Extract the job result and cost values
job_result = data["job_result"]
cost_values = data["cost_values"]


pub_result = job_result[0]
counts_bin = pub_result.data.meas.get_counts()
counts_int = pub_result.data.meas.get_int_counts()

shots = sum(counts_int.values())
final_distribution_int = {key: val for key, val in counts_int.items()}
final_distribution_bin = {key: val for key, val in counts_bin.items()}
print(final_distribution_bin)


#%% ------------- re-Generate the Graph  -------------

# Should be the same graph is initialy define
n = 5 # nodes number
graph = rx.PyGraph()
graph.add_nodes_from(np.arange(0, n, 1))
edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
graph.add_edges_from(edge_list)
draw_graph(graph, node_size=600, with_labels=True)


# Auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, len(graph))
most_likely_bitstring.reverse()

print("Result bitstring:", most_likely_bitstring)


#%%
# Plot results distribution
matplotlib.rcParams.update({"font.size": 10})
final_bits = final_distribution_bin
values = np.abs(list(final_bits.values()))
top_4_values = sorted(values, reverse=True)[:4]

positions = []
for value in top_4_values:
    positions.append(np.where(values == value)[0])
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1)
plt.xticks(rotation=45)
plt.title("Result Distribution")
plt.xlabel("Bitstrings (reversed)")
plt.ylabel("Probability")
ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")

for p in positions:
    ax.get_children()[int(p)].set_color("tab:purple")
plt.show()


#%%
# Visualize the best cut - auxiliary function to plot graphs
def plot_result(G, x):
    colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    pos, default_axes = rx.spring_layout(G), plt.axes(frameon=True)
    rx.visualization.mpl_draw(G, node_color=colors, node_size=200, alpha=0.8, pos=pos)

plot_result(graph, most_likely_bitstring)


from typing import Sequence
def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(x[u] * (1 - x[v]) + x[v] * (1 - x[u]) for u, v in list(graph.edge_list()))


cut_value = evaluate_sample(most_likely_bitstring, graph)
print('The value of the cut is:', cut_value)

print('Done.')
# %%
