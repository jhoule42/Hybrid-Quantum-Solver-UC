# QAOA Unbalence Knpasack Implementation
# Note: Only compatible with python version between 3.8 and 3.10

#%% -----------------------------   QAOA circuit ------------------------------------
from collections import defaultdict
import pennylane as qml
import numpy as np

#%%

items_values = {"⚽️": 8, "💻": 47, "📸": 10, "📚": 5, "🎸": 16}
values_list = [8, 47, 10, 5, 16]
items_weight = {"⚽️": 3, "💻": 11, "📸": 14, "📚": 19, "🎸": 5}
weights_list = [3, 11, 14, 19, 5]
maximum_weight = 26

#%%
import numpy as np

def sum_weight(bitstring, items_weight):
    weight = 0
    for n, i in enumerate(items_weight):
        if bitstring[n] == "1":
            weight += i
    return weight


def sum_values(bitstring, items_value):
    value = 0
    for n, i in enumerate(items_value):
        if bitstring[n] == "1":
            value += i
    return value

items = list(items_values.keys())
n_items = len(items)
combinations = {}
max_value = 0
for case_i in range(2**n_items):  # all possible options
    combinations[case_i] = {}
    bitstring = np.binary_repr(
        case_i, n_items
    )  # bitstring representation of a possible combination, e.g, "01100" in our problem means bringing (-💻📸--)
    combinations[case_i]["items"] = [items[n] for n, i in enumerate(bitstring) if i == "1"]
    combinations[case_i]["value"] = sum_values(bitstring, values_list)
    combinations[case_i]["weight"] = sum_values(bitstring, weights_list)
    # save the information of the optimal solution (the one that maximizes the value while respecting the maximum weight)
    if (
        combinations[case_i]["value"] > max_value
        and combinations[case_i]["weight"] <= maximum_weight
    ):
        max_value = combinations[case_i]["value"]
        optimal_solution = {
            "items": combinations[case_i]["items"],
            "value": combinations[case_i]["value"],
            "weight": combinations[case_i]["weight"],
        }


print(
    f"The best combination is {optimal_solution['items']} with a total value: {optimal_solution['value']} and total weight {optimal_solution['weight']} "
)

#%% -------------------------------------------------------





shots = 5000  # Number of samples used
dev = qml.device("default.qubit", shots=shots)


@qml.qnode(dev)
def qaoa_circuit(gammas, betas, h, J, num_qubits):
    wmax = max(
        np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
    )  # Normalizing the Hamiltonian is a good idea
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


def samples_dict(samples, n_items):
    """Just sorting the outputs in a dictionary"""
    results = defaultdict(int)
    for sample in samples:
        results["".join(str(i) for i in sample)[:n_items]] += 1
    return results


#%%

import matplotlib.pyplot as plt
# Annealing schedule for QAOA
betas = np.linspace(0, 1, 10)[::-1]  # Parameters for the mixer Hamiltonian
gammas = np.linspace(0, 1, 10)  # Parameters for the cost Hamiltonian (Our Knapsack problem)

fig, ax = plt.subplots()
ax.plot(betas, label=r"$\beta_i$", marker="o", markersize=8, markeredgecolor="black")
ax.plot(gammas, label=r"$\gamma_i$", marker="o", markersize=8, markeredgecolor="black")
ax.set_xlabel("i", fontsize=18)
ax.legend()
fig.show()
# %%

def from_Q_to_Ising(Q, offset):
    """Convert the matrix Q of Eq.3 into Eq.13 elements J and h"""
    n_qubits = len(Q)  # Get the number of qubits (variables) in the QUBO matrix
    # Create default dictionaries to store h and pairwise interactions J
    h = defaultdict(int)
    J = defaultdict(int)

    # Loop over each qubit (variable) in the QUBO matrix
    for i in range(n_qubits):
        # Update the magnetic field for qubit i based on its diagonal element in Q
        h[(i,)] -= Q[i, i] / 2
        # Update the offset based on the diagonal element in Q
        offset += Q[i, i] / 2
        # Loop over other qubits (variables) to calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Update the pairwise interaction strength (J) between qubits i and j
            J[(i, j)] += Q[i, j] / 4
            # Update the magnetic fields for qubits i and j based on their interactions in Q
            h[(i,)] -= Q[i, j] / 4
            h[(j,)] -= Q[i, j] / 4
            # Update the offset based on the interaction strength between qubits i and j
            offset += Q[i, j] / 4
    # Return the magnetic fields, pairwise interactions, and the updated offset
    return h, J, offset
    

#%%
def energy_Ising(z, h, J, offset):
    """
    Calculate the energy of an Ising model given spin configurations.

    Parameters:
    - z: A dictionary representing the spin configurations for each qubit.
    - h: A dictionary representing the magnetic fields for each qubit.
    - J: A dictionary representing the pairwise interactions between qubits.
    - offset: An offset value.

    Returns:
    - energy: The total energy of the Ising model.
    """
    if isinstance(z, str):
        z = [(1 if int(i) == 0 else -1) for i in z]
    # Initialize the energy with the offset term
    energy = offset
    # Loop over the magnetic fields (h) for each qubit and update the energy
    for k, v in h.items():
        energy += v * z[k[0]]
    # Loop over the pairwise interactions (J) between qubits and update the energy
    for k, v in J.items():
        energy += v * z[k[0]] * z[k[1]]
    # Return the total energy of the Ising model
    return energy


#%%
Q = -np.diag(list(items_values.values()))  # Matrix Q for the problem above.
x_opt = np.array(
    [[1 if i in optimal_solution["items"] else 0] for i in items_values.keys()]
)  # Optimal solution.
opt_str = "".join(str(i[0]) for i in x_opt)
min_cost = (x_opt.T @ Q @ x_opt)[0, 0]  # using Equation 3 above
print(f"Q={Q}")
print(f"The minimum cost is  {min_cost}")



#%%

N = round(np.ceil(np.log2(maximum_weight)))  # number of slack variables
weights = list(items_weight.values()) + [2**k for k in range(N)]

QT = np.pad(Q, ((0, N), (0, N)))  # adding the extra slack variables at the end of the Q matrix
n_qubits = len(QT)
lambd = 2  # We choose a lambda parameter enough large for the constraint to always be fulfilled
# Adding the terms for the penalty term
for i in range(len(QT)):
    QT[i, i] += lambd * weights[i] * (weights[i] - 2 * maximum_weight)  # Eq. 10
    for j in range(i + 1, len(QT)):
        QT[i, j] += 2 * lambd * weights[i] * weights[j]  # Eq. 9
offset = lambd * maximum_weight**2
print(f"Q={QT}")
# optimal string slack string
slack_string = np.binary_repr(maximum_weight - optimal_solution["weight"], N)[::-1]
x_opt_slack = np.concatenate(
    (x_opt, np.array([[int(i)] for i in slack_string]))
)  # combining the optimal string and slack string
opt_str_slack = "".join(str(i[0]) for i in x_opt_slack)
cost = (x_opt_slack.T @ QT @ x_opt_slack)[0, 0] + offset  # Optimal cost using equation 3
print(f"Cost:{cost}")

# At this point, we have encoded the problem in a format that we can use to solve it on quantum
# computers. Now it only remains to solve it using quantum algorithms!


#%%
# Our previous example should give us the same result
z_exp = [(1 if i == 0 else -1) for i in x_opt_slack]  # Converting the optimal solution from (0,1) to (1, -1)
h, J, zoffset = from_Q_to_Ising(QT, offset)  # Eq.13 for our problem

energy = energy_Ising(z_exp, h, J, zoffset)  # Caluclating the energy (Should be the same that for the QUBO)
print(f"Minimum energy:{energy}")

samples_slack = samples_dict(qaoa_circuit(gammas, betas, h, J, num_qubits=len(QT)), n_qubits)
values_slack = {sum_values(sample_i, values_list): count
    for sample_i, count in samples_slack.items()
        if sum_weight(sample_i, weights_list) <= maximum_weight}  # saving only the solutions that fulfill the constraint

print(f"The number of optimal solutions using slack variables \
        is {samples_slack[opt_str_slack]} out of {shots}")



# %% ----------- Unconstrained QAOA -----------

from openqaoa.problems import FromDocplex2IsingModel
from docplex.mp.model import Model


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
mdl = Knapsack(values_list, weights_list, maximum_weight)
lambda_1, lambda_2 = (0.96, 0.0371,)  # Params of the unbalanced penalization function (from the main paper)

ising_hamiltonian = FromDocplex2IsingModel(mdl, unbalanced_const=True,
                                        strength_ineq=[lambda_1, lambda_2]).ising_model

h_new = {tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 1}
J_new = {tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 2}

samples_unbalanced = samples_dict(qaoa_circuit(gammas, betas, h_new, J_new, num_qubits=n_items), n_items)

values_unbalanced = {sum_values(sample_i, values_list): count
    for sample_i, count in samples_unbalanced.items()
        if sum_weight(sample_i, weights_list) <= maximum_weight}  # saving only the solutions that fulfill the constraint

print(f"The number of solutions using unbalanced penalization is {samples_unbalanced[opt_str]} out of {shots}")
# %%
