#%%
import random
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import itertools
from tqdm import tqdm

import sys
sys.path.append("/Users/julien-pierrehoule/Documents/Stage/T3/Code")

#%%
def generate_knapsack_items(n, max_value=100, max_weight=10):
    """
    Generates items for a knapsack problem.
    
    Parameters:
    n (int): Number of items to generate.
    max_value (int): Maximum value for an item.
    max_weight (int): Maximum weight for an item.
    """
    values = [random.randint(1, max_value) for _ in range(n)]
    weights = [random.randint(1, max_weight) for _ in range(n)]
    return values, weights


def grid_search(v, w, c, k, theta, N_beta, N_gamma, mixer):
    """
    Algorithm 8: βγ-GridSearch for initial optimization of β and γ.

    Parameters:
    - QKP_func: Function that takes (beta, gamma) and returns the expected value.
    - N_beta: Number of grid points for beta.
    - N_gamma: Number of grid points for gamma.
    - beta_range: Tuple (min_beta, max_beta) defining the range for beta.
    - gamma_range: Tuple (min_gamma, max_gamma) defining the range for gamma.

    Returns:
    - (best_beta, best_gamma): Tuple of optimized beta and gamma values.
    """
    best_value = -np.inf
    best_beta, best_gamma = None, None
    
    # Create grid ranges for beta and gamma
    beta_values = [np.pi * i / N_beta for i in range(N_beta)]
    gamma_values = [2 * np.pi * j / N_gamma for j in range(N_gamma)]
    
    for beta in beta_values:
        for gamma in gamma_values:

            # Evaluate the QKP function at each (beta, gamma)
            value = QKP(v, w, c, beta, gamma, k, theta, mixer)
            
            # Update the best parameters if a better value is found
            if value > best_value:
                print(f"New best solution: {value}, {beta}, {gamma}")
                best_value = value
                best_beta = beta
                best_gamma = gamma
    
    return best_beta, best_gamma, best_value


def QKP_value(v, w, c, beta, gamma, k, theta, mixer):
    """ QKP Wrapper that only return the value. Usefull for minimization process. """
    bitstring, value, counts = QKP(v, w, c, beta, gamma, k, theta, mixer)
    return value


def parameter_optimization(v, w, c, k_range, theta_range, N_beta=50, N_gamma=50, mixer='copula'):
    """
    Algorithm 7: Parameter Optimization using a combination of grid search and BFGS optimization.

    Parameters:
    - QKP_func: Function that takes (beta, gamma, k, theta) and returns the expected value.
    - k_range: List of k values to try.
    - theta_range: List of theta values to try (use None if theta is not needed).
    - N_beta: Number of grid points for beta in the grid search.
    - N_gamma: Number of grid points for gamma in the grid search.

    Returns:
    - (best_params, best_value): Tuple containing the best parameter set and its corresponding value.
    """
    best_value = -np.inf
    best_params = None

    for k in k_range:
        for theta in theta_range:
            print(f"\nParameters (k, θ): {int(k), theta}" )

            # Step 1: Perform a grid search to get initial values of beta and gamma
            beta0, gamma0, value = grid_search(v, w, c, k, theta, N_beta, N_gamma, mixer)

            if value > best_value:
                best_value = value
                best_params = (beta0, gamma0, k, theta)

            # Step 2: Use BFGS for fine-tuning
            def objective(params):
                beta, gamma = params
                return -QKP_value(v, w, c, beta, gamma, k, theta, mixer)  # Negative for maximization

            # BFGS optimization starting from the best grid search result
            try:
                result = minimize(objective, [beta0, gamma0], method='BFGS')

                if result.success:
                    optimized_value = -result.fun
                    if optimized_value > best_value:
                        best_value = optimized_value
                        best_params = (result.x[0], result.x[1], k, theta)
                    print("Results:", result)

            except Exception as e:
                print(f"BFGS optimization failed for k = {k}, theta = {theta} with error: {e}")
                continue  # Skip to the next iteration if optimization fails
    
    return best_params, best_value



def logistic_bias(v, w, c, k):
    """ Creates a biased initial distribution using the logistic function for the Knapsack Problem."""
    r = np.array(v) / np.array(w)  # Calculate efficiency ratios
    C = (sum(w) / c) - 1
    # r_star = 0.6
    return 1 / (1 + C * np.exp(-k * (r - r.mean())))


def apply_cost_unitary(qc, gamma, values):
    """ Applies the cost unitary UC(γ) to the quantum circuit."""
    for i, v in enumerate(values):
        qc.rz(-2 * gamma * v, i) # Equivalent to a phase shift based on item value


def apply_hourglass_mixer(qc, beta, p):
    """ Applies the Hourglass mixer UBZX(β) using the biased probabilities p."""
    for i, pi in enumerate(p):
        angle = 2 * np.arcsin(np.sqrt(pi))
        qc.ry(2 * angle, i) # RY gate to create initial bias
        qc.rz(-2 * beta, i)
        qc.ry(-2 * angle, i) # Undo the bias


def apply_X_mixer(qc, beta):
    """ Applies the standard QAOA X mixer UB(β).
    
    Args:
        qc (QuantumCircuit): The quantum circuit to apply the mixer to
        beta (float): The mixing angle parameter
        
    The X mixer applies RX rotations to all qubits, implementing the mixing
    Hamiltonian HB = sum_j X_j where X_j is the Pauli X operator on qubit j.
    """
    num_qubits = qc.num_qubits
    for i in range(num_qubits):
        qc.rx(2 * beta, i)  # Apply RX rotation with angle 2β to each qubit


def apply_copula_mixer(qc, beta, p1, p2, theta):
    """ Applies the two-qubit Copula mixer UCop12 with correlation parameter θ. """
    for i in range(len(p1) - 1):
        phi1 = 2 * np.arcsin(np.sqrt(p1[i]))
        phi2 = 2 * np.arcsin(np.sqrt(p2[i]))
        
        qc.ry(phi1, i)
        qc.ry(phi2, i + 1)
        qc.cz(i, i + 1)  # Controlled phase to introduce correlation
        qc.ry(-phi1, i)
        qc.ry(-phi2, i + 1)
        qc.rz(-2 * beta, i)
        qc.rz(-2 * beta, i + 1)


def QKP(v, w, c, gamma, beta, k, theta=None, mixer='hourglass', verbose=False):
    """
    Algorithm 5: Solving the Knapsack Problem using Quantum Optimization (QAOA)
    
    Parameters:
    - n: Number of items.
    - values: List of item values.
    - weights: List of item weights.
    - capacity: Knapsack capacity.
    - gamma: Phase shift parameter.
    - beta: Mixing parameter.
    - k: Bias strength for logistic distribution.
    - theta: Correlation parameter for the Copula mixer (if used).
    - mixer: Type of mixer to use ('hourglass' or 'copula').
    
    Returns:
    - Best solution found among multiple samples.
    """
    n = len(v)

    # Initialize biased probabilities
    p = logistic_bias(v, w, c, k)

    # Create a quantum circuit with n qubits
    qc = QuantumCircuit(n)
    
    # Apply initial biased state preparation
    for i in range(n):
        angle = 2 * np.arcsin(np.sqrt(p[i]))
        qc.ry(angle, i)
    qc.barrier()
    
    # Apply cost unitary
    apply_cost_unitary(qc, gamma, v)
    qc.barrier()
    
    # Apply mixer
    if mixer == 'X':
        apply_X_mixer(qc, beta)

    elif mixer == 'hourglass':
        apply_hourglass_mixer(qc, beta, p)

    elif mixer == 'copula' and theta is not None:
        p2 = logistic_bias(v, w, c, k)
        apply_copula_mixer(qc, beta, p, p2, theta)

    # Measurement
    qc.measure_all()


    # Instantiate a new statevector simulation based sampler object.
    sampler = StatevectorSampler()
 
    # pt moyen de metre des params en imput sur lequel sweep
    pub = (qc)
    job = sampler.run([pub], shots=10000)

    # Extract the result for the 0th pub (this example only has one pub).
    result = job.result()[0]
    bitstrings = result.data.meas.get_counts()

    # Find the best solution based on counts
    best_solution = max(bitstrings, key=bitstrings.get)
    best_value = sum(int(best_solution[i]) * v[i] for i in range(n))
    total_weight = sum(int(best_solution[i]) * w[i] for i in range(n))


    if verbose:
        print("Counts", bitstrings)
        print(f"Best solution: {best_solution}")
        print(f"Best values: {best_value}")
        print(f"total Weight: {total_weight}")

    # Check if the solution is valid
    # if total_weight <= capacity:
    return best_solution, float(best_value), bitstrings
    # else:
    #     return None, 0, None  # Invalid solution due to weight constraint


def bruteforce_knapsack(values, weights, capacity):
    """
    Brute-force solver for the knapsack problem with a progress bar.

    Parameters:
    - values (list): List of item values.
    - weights (list): List of item weights.
    - capacity (int): Maximum capacity of the knapsack.

    Returns:
    - ranked_solutions (list): List of tuples (value, weight, items) sorted by value in descending order.
    """
    n = len(values)
    ranked_solutions = []
    
    # Total number of combinations to evaluate
    total_combinations = 2 ** n

    # Generate all possible combinations of items with a progress bar
    for subset in tqdm(itertools.product([0, 1], repeat=n), total=total_combinations, desc="Evaluating knapsack combinations"):
        # Calculate total weight and value of the current subset
        total_weight = sum(weights[i] * subset[i] for i in range(n))
        total_value = sum(values[i] * subset[i] for i in range(n))
        
        # Check if the subset fits within the capacity
        if total_weight <= capacity:
            ranked_solutions.append((total_value, total_weight, subset))

        else:
            ranked_solutions.append((0, 0, subset))

    
    # Sort the solutions by total value in descending order
    ranked_solutions.sort(key=lambda x: x[0], reverse=True)

    ranked_solutions = [(int(value), int(weight), ''.join(map(str, bitstring))) \
                    for value, weight, bitstring in ranked_solutions]

    return ranked_solutions


