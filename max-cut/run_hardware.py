#%%####################################################################
#               Running QAOA for Max-Cut on real Hardware           #
#####################################################################
"""Run Max-Cut on real hardware."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import qpy
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import Session, Options
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

#%%
# qc should normaly be optimised for the quantum computer used
# here we only use the least busy one
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(operational=True, simulator=False)


#%% -------------------- Load quantum circuit --------------------
with open('circuits/test.qpy', 'rb') as handle:
    qc = qpy.load(handle)
 
qc[0].draw('mpl')
qc[0].decompose().draw('mpl')
circuit = qc[0]
circuit.draw('mpl')

init_params = [np.pi,  np.pi/2, np.pi,  np.pi/2]

max_cut_paulis = [('IIIZZ',1.0), ('IIZIZ',1.0), ('ZIIIZ',1.0),
                  ('IIZZI',1.0), ('IZZII',1.0), ('ZZIII',1.0)]
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)


# # Transpiling the quantum circuit
# pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
# isa_ansatz = pm.run(circuit)
# # isa_observable = observable_2.apply_layout(layout = isa_ansatz.layout)



#%% QiskitRuntimeService.save_account(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>", overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(min_num_qubits=127)
print(backend)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3,
                                    backend=backend)

candidate_circuit = pm.run(circuit)
candidate_circuit.draw('mpl', fold=False, idle_wires=False)


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

#%%

from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize

objective_func_vals = [] # Global variable
with Session(backend=backend) as session:
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"

    result = minimize(
        cost_func,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    print(result)


# plotting cost value convergence
plt.figure(figsize=(12, 6))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()



#%%
# Once the optimal parameters have been found, we assign these parameters
# and sample the final distribution
optimized_circuit = candidate_circuit.assign_parameters(result.x)
optimized_circuit.draw('mpl', fold=False, idle_wires=False)



# Run Sample
sampler = Sampler(mode=backend)
sampler.options.default_shots = 1000

# Set simple error suppression/mitigation options
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XY4"
sampler.options.twirling.enable_gates = True
sampler.options.twirling.num_randomizations = "auto"

pub = (optimized_circuit, )
job = sampler.run([pub], shots=int(1e4))

counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
print(final_distribution_int)



job_id = "cwdx4tb9r49g0085h73g"
job = service.job(job_id)

exp_val_list = job.result()[0].data.evs
