""" Classical Solver used for to solve the Unit Commtiment problem.
Author: Julien-Pierre Houle """
#%%
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize, BFGS


def classical_unit_commitment_qp(A, B, C, L, p_min, p_max, verbose=False):
    """
    Solve the Unit Commitment problem using a quadratic programming solver (Gurobi).
    
    Args:
        A (list): Fixed cost coefficients for each unit.
        B (list): Linear operational cost coefficients for each unit.
        C (list): Quadratic operational cost coefficients for each unit.
        L (float): Total power demand (load).
        p_min (list): Minimum power output for each unit.
        p_max (list): Maximum power output for each unit.

    Returns:
        y_solution (list): Binary solution (on/off for each unit).
        p_solution (list): Power output solution for each unit.
        total_cost (float): Total cost of operation.
    """
    
    n_units = len(A)  # Number of power units

    # Create a new model
    model = gp.Model("unit_commitment")

    # Suppress output if verbose is False
    if not verbose:
        model.setParam('OutputFlag', 0)

    # Add binary decision variables for turning the units on/off
    y = model.addVars(n_units, vtype=GRB.BINARY, name="y")

    # Add continuous decision variables for the power output of each unit
    p = model.addVars(n_units, lb=0, name="p")

    # Objective function: minimize total cost
    total_cost = gp.quicksum( (A[i]*y[i]) + (B[i]*p[i]) + (C[i]*p[i]*p[i]) for i in range(n_units))
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Add constraint: Total power output must meet the load demand L
    model.addConstr(gp.quicksum(p[i] for i in range(n_units)) == L, name="power_balance")

    # Add constraints: Power output of each unit must respect on/off state and power bounds
    for i in range(n_units):
        model.addConstr(p[i] >= p_min[i] * y[i], name=f"min_power_{i}")
        model.addConstr(p[i] <= p_max[i] * y[i], name=f"max_power_{i}")

    # Increase precision in Gurobi
    model.setParam('MIPGap', 1e-12)  # Tighter gap for optimality
    model.setParam('FeasibilityTol', 1e-6)  # Tighter feasibility tolerance

    # Optimize the model
    model.optimize()


    if model.status == GRB.OPTIMAL:
        # Extract binary (y) and continuous (p) solutions
        y_solution = [y[i].x for i in range(n_units)]
        p_solution = [p[i].x for i in range(n_units)]
        total_cost = model.objVal

        y_solution = ''.join(['1' if num == 1.0 else '0' for num in y_solution]) # convert to string

        return y_solution, p_solution, total_cost, model.Runtime
    else:
        raise ValueError("Optimization failed")
    



def classical_power_distribution(x_sol, A, B, C, p_min, p_max, L, raise_error=True):
    """
    Distribute power among active units based on the binary solution from the QUBO problem.

    Args:
        x_sol (str): Binary solution string (from QAOA).
        B (list): Linear power coefficients for each unit.
        C (list): Quadratic power coefficients for each unit.
        p_min (list): Minimum power output for each unit.
        p_max (list): Maximum power output for each unit.
        L (float): Required total power load.

    Returns:
        tuple: Optimal power outputs and the associated cost.
    """

    nb_units = len(x_sol)
    active_units = [i for i, bit in enumerate(x_sol) if bit == '1']

    # Raise Value Errors if the optimisation is impossible
    if sum(p_max[i] for i in active_units) < L:
        if raise_error:
            raise ValueError("Total maximum power output of active units is less than required load L.")
        else:
            return [], 0
        
    if min(p_min) > L:
        if raise_error:
            raise ValueError("Minimum power output is more than the requiered load L.")
        else:
            return [], 0        

    if not active_units:
        if raise_error:
            raise ValueError("No active units, cannot distribute power.")
        return [], 0

    # Objective function: Minimize power generation cost for active units
    def objective(power):
        """Objective cost function to minimize."""
        cost = 0
        for i in active_units:
            cost += A[i] + (B[i]*power[i]) + (C[i]*(power[i]**2))
        return cost

    # Constraint to ensure total power output meets the load L
    def load_constraint(p):
        return np.sum(p) - L

    # Define bounds for power outputs
    bounds = [(p_min[i], p_max[i]) if x_sol[i] == '1' else (0, 0) for i in range(nb_units)]

    # Initial guess: even distribution of L among active units
    num_active_units = len(active_units)
    initial_guess = [0] * nb_units
    if num_active_units > 0:
        even_distribution = L / num_active_units
        for i in active_units:
            initial_guess[i] = min(max(even_distribution, p_min[i]), p_max[i])

    # Optimization with constraints
    constraints = [{'type': 'eq', 'fun': load_constraint}]
    
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    # Check if the total power distribution matches the load L
    total_power = np.sum(result.x)
    if np.abs(total_power - L) > 1e-5:
        print(f"Warning: Total power distribution {total_power} does not match the load L={L}")

    return result.x, result.fun  # Return optimal power outputs and cost




#%% ---------------------- PLOT GRAPH ----------------------

# # Classical Solving Time by number of Units
# nb_units_range = np.arange(10, 1000, 10)
# runtime_gurobi_arr = np.zeros(nb_units_range.size)

# for idx, nb_units in enumerate(nb_units_range):
#     A, B, C, p_min, p_max = generate_units(N=nb_units)

#     max_power = np.sum(p_max)
#     min_power = np.min(p_min)
#     L = np.random.uniform(min_power*1.2, max_power*0.9)

#     y_solution, p_solution, total_cost, runtime = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)

#     runtime_gurobi_arr[idx] = runtime

# # %%

# # Plotting the runtime for Gurobi
# plt.figure(figsize=(10, 6))
# plt.plot(nb_units_range, runtime_gurobi_arr, label='Gurobi Solver', marker='o', linestyle='-', color='blue')

# # Adding title and labels
# plt.title('Gurobi Solver Runtime vs Number of Units - UC problem')
# plt.xlabel('Number of Units')
# plt.ylabel('Runtime (sec)')

# # Adding grid, legend, and aesthetics
# plt.grid(True)
# plt.legend(loc='upper left')
# plt.yscale('log')

# # Show the plot
# plt.savefig('Figures/gurobi_solver_runtime.png', bbox_inches='tight')
# plt.close()

# %%