#%% Code use to generate the Hamiltonian with Unbalanced Penalization
import json
import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")
from xQAOA.kp_utils import *
from qiskit.visualization import plot_histogram, plot_distribution
from openqaoa.problems import FromDocplex2IsingModel
from docplex.mp.model import Model

#%% =================== PROBLEM PARAMETERS ===================

PATH_TO_SAVE = "/Users/julien-pierrehoule/Documents/Stage/T3/Code/xQAOA"

values, weights = generate_profit_spanner(n=10)
c = np.ceil(0.6 * sum(weights)).astype(int)

# items_values = {"⚽️": 8, "💻": 47, "📸": 10, "📚": 5, "🎸": 16}
# values_list = [8, 47, 10, 5, 16]
# items_weight = {"⚽️": 3, "💻": 11, "📸": 14, "📚": 19, "🎸": 5}
# weights_list = [3, 11, 14, 19, 5]
# maximum_weight = 26


def Knapsack(values, weights, maximum_weight):
    """Create a docplex model of the problem. (Docplex is a classical solver from IBM)"""
    n_items = len(values)
    mdl = Model()
    x = mdl.binary_var_list(range(n_items), name="x")
    cost = -mdl.sum(x[i] * values[i] for i in range(n_items))
    mdl.minimize(cost)
    mdl.add_constraint(mdl.sum(x[i] * weights[i] for i in range(n_items)) <= maximum_weight)
    return mdl


# Docplex model, we need to convert our problem in this format to use the unbalanced penalization approach
mdl = Knapsack(values, weights, c)
lambda_1, lambda_2 = (0.9603, 0.0371)  # Parameters of the unbalanced penalization function (see the main paper)

# Creating the Hamiltonian
ising_hamiltonian = FromDocplex2IsingModel(mdl,
                                           unbalanced_const=True,
                                           strength_ineq=[lambda_1, lambda_2]
                                           ).ising_model  # https://arxiv.org/abs/2211.13914

# Convert tuple keys to strings
h_new = {str(tuple(i)): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 1}
J_new = {str(tuple(i)): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 2}

# Prepare data to save
data_to_save = {
    "items_values": list(map(int, weights)),
    "items_weights": list(map(int, weights)),
    "capacity": float(c),
    "h_terms": h_new,
    "j_terms": J_new,
}

# Save to a JSON file
output_file = "knapsack_data.json"
with open(f"{PATH_TO_SAVE}/hamiltonian_file.json", "w") as file:
    json.dump(data_to_save, file, indent=4)

print(f"H terms: {h_new}")
print(f"J terms: {J_new}")
print(f"Data saved to {output_file}")

# %%
