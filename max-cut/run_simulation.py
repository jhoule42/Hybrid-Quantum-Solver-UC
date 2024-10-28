# Run local simulation of quantum circuits to solve max-cut.

#%%
import json
import time
import numpy as np
from numpy.linalg import eigvalsh
from scipy.optimize import minimize

from qiskit import qpy
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_ibm_runtime import RuntimeEncoder
from qiskit_ibm_runtime import RuntimeDecoder


#%% -------------------- Load quantum circuit --------------------
with open('circuits/test.qpy', 'rb') as handle:
    qc = qpy.load(handle)
 
qc[0].draw('mpl')
qc[0].decompose().draw('mpl')
circuit = qc[0]

init_params = [np.pi,  np.pi/2, np.pi,  np.pi/2]

max_cut_paulis = [('IIIZZ',1.0), ('IIZIZ',1.0), ('ZIIIZ',1.0),
                  ('IIZZI',1.0), ('IZZII',1.0), ('ZZIII',1.0)]
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)



cost_vals = []

def cost_func(params, ansatz, hamiltonian, estimator):
    """ Return estimate of energy from estimator.

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs
    cost_vals.append(cost)

    return cost


#%% -------------------- Minimisation routine --------------------

print("Minimisation routine.")

start_time = time.time()

# The circuit needs to be transpiled to the AerSimulator target
pass_manager = generate_preset_pass_manager(3, AerSimulator())
isa_circuit = pass_manager.run(circuit) # instruction set architecture circuit

exact_estimator = Estimator()
result = minimize(cost_func,
                  x0 = init_params,
                  args=(isa_circuit, cost_hamiltonian, exact_estimator),
                  method="COBYLA",
                  options={'maxiter': 1000, 'disp': False})

end_time = time.time()
execution_time = end_time - start_time
print(result)

# Classical comparison using NumPy
print("Classical computation using Numpy.")
sol_numpy = min(eigvalsh(cost_hamiltonian.to_matrix()))
print(f"Number of iterations: {result.nfev}") # nb times func is call
print(f"Time(s): {execution_time:.4f}")
print(f"Percent error: {abs((result.fun - sol_numpy)/sol_numpy):.2e}")


#%%
# plotting cost value convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(cost_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()



#%% ------------------------- Sampling -------------------------
# Once the optimal parameters have been found, we assign these parameters

exact_sampler = Sampler()
optimized_circuit = isa_circuit.assign_parameters(result.x)
optimized_circuit.draw('mpl', fold=False, idle_wires=False)

measured_circuit = isa_circuit.copy()
measured_circuit.measure_all() # adding measurments

# The circuit needs to be transpiled to the AerSimulator target
pass_manager = generate_preset_pass_manager(3, AerSimulator())
isa_circuit = pass_manager.run(measured_circuit)
pub = (isa_circuit, result.x, 1000)
job = exact_sampler.run([pub])
job_result = job.result()
pub_result = job_result[0]

counts = pub_result.data.meas.get_counts()
plot_histogram(counts, title='Sample')


#%% ------------- Save Results to File -------------

# Create a dictionary to store both the job result and the cost values
data = {"job_result": job_result,
        "cost_values": np.array(cost_vals)}

with open("results/result.json", "w") as file:
    json.dump(data, file, cls=RuntimeEncoder)

print('Done.')
# %%
