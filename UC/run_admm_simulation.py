""" Code to run the ADMM algorithm."""
#%%
import numpy as np
from docplex.mp.model import Model
import pickle

from qiskit.primitives import Sampler
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer, ADMMOptimizer, ADMMParameters
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

from utils import *
from classical_solver_UC import *
from admm_optimizer import ADMMParameters, ADMMOptimizer

#%% ============================ Unit Commitment Model ============================

def UC_model(A, B, C, L, p_min, p_max):
    n_units = len(A)  # Number of power units
    mdl = Model("unit_commitment")
    
    # Decision variables
    y = mdl.binary_var_list(n_units, name="y")
    p = [mdl.continuous_var(lb=0, ub=p_max[i], name=f"p_{i}") for i in range(n_units)]
    
    # Objective function
    total_cost = mdl.sum((A[i]*y[i]) + (B[i]*p[i]) + (C[i]*(p[i] ** 2)) for i in range(n_units))
    mdl.minimize(total_cost)

    # Constraints
    mdl.add_constraint(mdl.sum(p) == L, "power_balance")
    epsilon = 0  # Small tolerance
    for i in range(n_units):
        mdl.add_constraint(p[i] >= p_min[i] * y[i] - epsilon, f"min_power_{i}")
        mdl.add_constraint(p[i] <= p_max[i] * y[i] + epsilon, f"max_power_{i}")

    # 3. At least one unit must be on (if required)
    # mdl.add_constraint(mdl.sum(y) >= 1, "Min_One_Active")

    qp = from_docplex_mp(mdl)
    print(qp.prettyprint())
    return qp


L = 80  # Total power demand
n_units = 3
A, B, C, p_min, p_max = generate_units(N=n_units)
param_exec = {"L": L,
              "n_units":n_units,
              "A": A,
              "B": B,
              "C": C,
              "p_min": p_min,
              "p_max": p_max}

# Generate quadratic program
qp_UC = UC_model(A, B, C, L, p_min, p_max)


#%% ======================= Classical Solver Gurobi =======================

print("\nGurobi Solver")
bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime = classical_unit_commitment_qp(A, B, C, L, p_min, p_max)
result_gurobi = [bitstring_gurobi, power_distrib_gurobi, cost_gurobi, runtime]
print(f"Bitstring: {bitstring_gurobi}")
print(f"Power distribution: {power_distrib_gurobi}")
print(f"Total Power Load: {np.sum(power_distrib_gurobi):.1f} | {L}")
print(f"Cost: {cost_gurobi:.2f}")


#%% ========================= ADMM CLASSICAL QUBO SOLVER =========================

# Set parameters
admm_params = ADMMParameters(rho_initial=2500,
                             beta=1000,
                             factor_c=100,
                             maxiter=100,
                             three_block=False,
                             tol=1e-12,
                             warm_start=True,
                             p=3)

# CLASSICAL SOLVER
admm_CS = ADMMOptimizer(params=admm_params, qubo_type='classical')
result_CS = admm_CS.solve(qp_UC)


x_sol = result_CS.x[:n_units] # extract binary string
power_distrib = result_CS.x[n_units:] # extract power distribution

# Set small values (below threshold) to zero
threshold = 1e-1
power_distrib = np.where(np.abs(power_distrib) < threshold, 0, power_distrib)

print("\nADMM Classical QUBO")
print(f"Cost: {result_CS.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_distrib}")
print(f"Total Power Load: {np.sum(power_distrib):.1f} | {L}")



#%% ========================= ADMM BASIC QUANTUM SOLVER =========================

param_history = []
value_history = []
eval_count_history = []

def callback(eval_count, params, mean, std):
    """Callback to capture optimization progress."""
    param_history.append(params)
    value_history.append(mean)


def extract_parameters(param_history, p):
    """Function to extract gamma and beta values for each step."""
    gammas_history = []  # List to store all gamma values per step
    betas_history = []   # List to store all beta values per step

    for params in param_history:
        gammas = params[:p]      # First p values are gammas
        betas = params[p:2 * p]  # Next p values are betas

        gammas_history.append(gammas)
        betas_history.append(betas)

    return gammas_history, betas_history


# Set parameters
admm_params = ADMMParameters(rho_initial=2500,
                             beta=1000,
                             factor_c=100,
                             maxiter=100,
                             three_block=False,
                             tol=1e-12,
                             warm_start=True)

p = 3
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=p, callback=callback)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)

admm_BQ = ADMMOptimizer(params=admm_params,
                        qubo_optimizer=qaoa_optimizer,
                        qubo_type='qaoa_basic')
result_BQ = admm_BQ.solve(qp_UC)

x_sol = result_BQ.x[:n_units] # extract binary string
power_dist_BQ = result_BQ.x[n_units:] # extract power distribution
power_dist_BQ = np.where(np.abs(power_dist_BQ) < 1e-3, 0, power_dist_BQ)

print("\nADMM BASIC Quantum SOLVER")
print(f"Cost: {result_BQ.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_dist_BQ}")
print(f"Total Power Load: {np.sum(power_dist_BQ):.1f} | {L}")

# Extract the parameters from qaoa basic
gammas_history, betas_history = extract_parameters(param_history, p)

# After the optimization, sample the final bitstring distribution
final_params = param_history[-1]  # Get the final parameters after optimization

#%% ========================= ADMM ADVANCE QUANTUM SOLVER =========================

admm_params = ADMMParameters(rho_initial=2500,
                             beta=1000,
                             factor_c=100,
                             maxiter=100,
                             three_block=False,
                             tol=1e-12,
                             warm_start=True,
                             p=3)

admm_AQ = ADMMOptimizer(params=admm_params, qubo_type='qaoa_advance')
result_AQ = admm_AQ.solve(qp_UC)
print(result_AQ.prettyprint())


x_sol = result_AQ.x[:n_units] # extract binary string
power_dist_AQ = result_AQ.x[n_units:] # extract power distribution
power_dist_AQ = np.where(np.abs(power_dist_AQ) < 1e-3, 0, power_dist_AQ)

print("\nADMM Advance Quantum")
print(f"Cost: {result_AQ.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_dist_AQ}")
print(f"Total Power Load: {np.sum(power_dist_AQ):.1f} | {L}")

result_scipy = result_AQ.state.results_qaoa_optim

#%% DEBUGING
print(f"State x0 saved BQ: {result_BQ.state.x0_saved}")
print(f"State x0 saved AQ: {result_AQ.state.x0_saved}")



#%% ========================= ADMM ADVANCE QUANTUM SOLVER HARDWARE =========================

admm_params = ADMMParameters(rho_initial=2500,
                             beta=1000,
                             factor_c=100,
                             maxiter=100,
                             three_block=False,
                             tol=1e-12,
                             warm_start=True,
                             p=2)

admm_AQ = ADMMOptimizer(params=admm_params,
                        qubo_type='qaoa_advance',
                        hardware_execution=True)

result_AQ = admm_AQ.solve(qp_UC)
print(result_AQ.prettyprint())


x_sol = result_AQ.x[:n_units] # extract binary string
power_dist_AQ = result_AQ.x[n_units:] # extract power distribution
power_dist_AQ = np.where(np.abs(power_dist_AQ) < 1e-3, 0, power_dist_AQ)

print("\nADMM Advance Quantum")
print(f"Cost: {result_AQ.fval:.1f}")
print(f"Bitstring: {x_sol}")
print(f"Power distribution: {power_dist_AQ}")
print(f"Total Power Load: {np.sum(power_dist_AQ):.1f} | {L}")

result_scipy = result_AQ.state.results_qaoa_optim

#%% DEBUGING
print(f"State x0 saved BQ: {result_BQ.state.x0_saved}")
print(f"State x0 saved AQ: {result_AQ.state.x0_saved}")




#%% ======================= Saving Results =======================

def save_results(filename, **kwargs):
    """Save multiple variables to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(kwargs, file)
        print(f"Results saved.")

# Save all relevant data
run_description = "Run ADMM Simulation #1"

PATH_RESULTS = 'results'
save_results(f"{PATH_RESULTS}/admm_results2.pkl", 
             run_description=run_description,
             param_exec=param_exec,
             result_gurobi=result_gurobi,
             result_CS=result_CS,
             result_BQ=result_BQ,
             result_AQ=result_AQ, 
             value_history=value_history, 
             param_history=param_history, 
             gammas_history=gammas_history, 
             betas_history=betas_history, 
             power_dist_BQ=power_dist_BQ,
             )


#%% ======================= Plotting Results =======================

visualize_admm_details({'Advance Quantum': result_AQ,
                    'Basic Quantum': result_BQ,
                    'Classical Qubo': result_CS},
                    save_path="/Users/julien-pierrehoule/Documents/Stage/T3/Code/UC/Figures/ADMM",
                    combine_plots=True,
                    filename_suffix='_2')

#%%

p = admm_AQ._params.p
plot_qaoa_optimization_results(result_scipy,
                               gamma0=np.linspace(0, 2*np.pi, p),
                               beta0=np.linspace(np.pi, 0, p),
                               show_legend=False)

#%%
# Power distribution
plot_optimal_power_distribution(param_exec,
                                Gurobi = power_distrib_gurobi,
                                Advance_Quantum = power_dist_AQ,
                                ADMM_Classical = power_distrib,
                                Basic_Quantum = power_dist_BQ,)

p = admm_BQ._params.p
init_gamma = np.linspace(0, 2*np.pi, p)
init_beta = np.linspace(np.pi, 0, p)
plot_qaoa_parameter_evolution(gammas_history, betas_history, p, init_gamma, init_beta)

plot_value_history(value_history)





# Make a graph about the cost vs iteration and see when solution start to be valid
# plot_admm_cost(result_quantum, rho_init, beta, factor_c, maxiter, three_block, tol)



# # Ranking the bitstring
# # ATTENTION: l'algo fait d'autre calcul après le dernier QAOA
# # Aussi il minimise x* --> c'est pas clair que c'est équivalent à l'autre problème!
# counts = admm_AQ._state.counts
# count_perf, count_rank = evaluate_perf_algo(counts, A, B, C, p_min, p_max, L)
# plot_custom_histogram(counts,
#                       highlighted_outcome=bitstring_gurobi,
#                       max_bitstrings=32,
#                       bitstring_rankings=count_rank)

# %%
