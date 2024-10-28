""" 
This code use the Qiskit.Optimisation community package to solve Knapsack using QAOA.
Author: Julien-Pierre Houle
Last Update: September 2024
"""

#%%
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.converters import QuadraticProgramToQubo


#%% Define the Knapsack Problem
prob = Knapsack(values=[3, 4, 5, 6, 7],
                weights=[2, 3, 4, 5, 6],
                max_weight=10)
qp = prob.to_quadratic_program()
print(qp.prettyprint())


#%%# Classical solution of the problem using Numpy
meo = MinimumEigenOptimizer(min_eigen_solver=NumPyMinimumEigensolver())
result = meo.solve(qp)
print(result.prettyprint())
print("\nsolution:", prob.interpret(result))


#%% Running QAOA
meo = MinimumEigenOptimizer(min_eigen_solver=QAOA(reps=1, sampler=Sampler(), optimizer=COBYLA()))
result = meo.solve(qp)
print(result.prettyprint())
print("\nsolution:", prob.interpret(result))
print("\ntime:", result.min_eigen_solver_result.optimizer_time)

#%%
# If you want to check the actual Hamiltonian generated from your problem instance,
# you need to apply a converter as follows.
# the same knapsack problem instance as in the previous section

# intermediate QUBO form of the optimization problem
conv = QuadraticProgramToQubo()
qubo = conv.convert(qp)
print(qubo.prettyprint())

# qubit Hamiltonian and offset
op, offset = qubo.to_ising()
print(f"num qubits: {op.num_qubits}, offset: {offset}\n")
print(op)
# %%