import random
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_ibm_runtime import SamplerV2


class QKPOptimizer:
    def __init__(self, v, w, c, mixer, run_hardware=False, backend=None,
                 sampler=None, pass_manager=None):
        self.v = v
        self.w = w
        self.c = c
        self.mixer = mixer
        self.backend = backend
        self.run_hardware = run_hardware
        self.pass_manager = pass_manager

        
        self.n = len(v)
        self.best_bitstring = None
        self.best_value = -np.inf
        self.best_weight = -np.inf
        if sampler == None:
            self.sampler = StatevectorSampler()
        else:
            self.sampler = sampler
    

    def logistic_bias(self, k):
        """Creates a biased initial distribution using the logistic function."""
        r = np.array(self.v) / np.array(self.w)
        C = (sum(self.w) / self.c) - 1
        return 1 / (1 + C * np.exp(-k * (r - r.mean())))
    

    def apply_cost_unitary(self, qc, gamma):
        """Applies the cost unitary UC(γ) to the quantum circuit."""
        for i, value in enumerate(self.v):
            qc.rz(-2 * gamma * value, i)


    def apply_hourglass_mixer(self, qc, beta, p):
        """Applies the Hourglass mixer UBZX(β)."""
        for i, pi in enumerate(p):
            angle = 2 * np.arcsin(np.sqrt(pi))
            qc.ry(2 * angle, i)
            qc.rz(-2 * beta, i)
            qc.ry(-2 * angle, i)


    def apply_X_mixer(self, qc, beta):
        """Applies the standard QAOA X mixer."""
        for i in range(self.n):
            qc.rx(2 * beta, i)


    def apply_copula_mixer(self, qc, beta, p1, p2, theta):
        """Applies the two-qubit Copula mixer."""
        for i in range(len(p1) - 1):
            phi1 = 2 * np.arcsin(np.sqrt(p1[i]))
            phi2 = 2 * np.arcsin(np.sqrt(p2[i]))
            qc.ry(phi1, i)
            qc.ry(phi2, i + 1)
            qc.cz(i, i + 1)
            qc.ry(-phi1, i)
            qc.ry(-phi2, i + 1)
            qc.rz(-2 * beta, i)
            qc.rz(-2 * beta, i + 1)


    def QKP(self, beta, gamma, k, theta=None, bit_mapping='regular'):
        p = self.logistic_bias(k)
        qc = QuantumCircuit(self.n)

        # Initial state preparation
        for i in range(self.n):
            angle = 2 * np.arcsin(np.sqrt(p[i]))
            qc.ry(angle, i)
        qc.barrier()

        # Cost unitary
        self.apply_cost_unitary(qc, gamma)
        qc.barrier()

        # Mixer application
        if self.mixer == 'X':
            self.apply_X_mixer(qc, beta)
        elif self.mixer == 'hourglass':
            self.apply_hourglass_mixer(qc, beta, p)
        elif self.mixer == 'copula' and theta is not None:
            p2 = self.logistic_bias(k)
            self.apply_copula_mixer(qc, beta, p, p2, theta)

        qc.measure_all()

        # Run on real hardware
        if self.run_hardware:
            # Transpilation
            print('Transpilation...')
            isa_qc = self.pass_manager.run(qc)
            backend = self.backend

            print('Running on hardware...')
            job = self.sampler.run([isa_qc], shots=10)
            result = job.result()[0]
            print("Done.")

        # Run on simulator
        else:
            job = self.sampler.run([qc], shots=50000)
            result = job.result()[0]

        counts = result.data.meas.get_counts()

        # Find best solution
        best_solution = max(counts, key=counts.get)

        if bit_mapping == 'regular':
            best_value = sum(int(best_solution[i]) * self.v[i] for i in range(self.n))
            total_weight = sum(int(best_solution[i]) * self.w[i] for i in range(self.n))

        
        if bit_mapping == 'inverse':
            best_value = sum((1 - int(best_solution[i])) * self.v[i] for i in range(self.n))
            total_weight = sum((1 - int(best_solution[i])) * self.w[i] for i in range(self.n))


        # Make sure the solution is valid
        if total_weight <= self.c:
            return best_solution, float(best_value), total_weight, counts, True
        else:
            return best_solution, float(best_value), total_weight, counts, False
        

    def QKP_value_wrapper(self, beta, gamma, k, theta, bit_mapping):
        """Wrapper that tracks the best bitstring while returning only the value."""
        bitstring, value, weight, counts, valid = self.QKP(beta, gamma, k, theta, bit_mapping)
        if (value > self.best_value) and (valid==True):
            print(f"New best solution: {int(value)}")
            self.best_bitstring = bitstring
            self.best_value = value
            self.best_weight = weight
            self.best_params = (beta, gamma)
        return value


    def grid_search(self, k, theta, N_beta, N_gamma, bit_mapping):
        """Grid search for optimization of β and γ."""
        best_value = -np.inf
        
        beta_values = [np.pi * i / N_beta for i in range(N_beta)]
        gamma_values = [2 * np.pi * j / N_gamma for j in range(N_gamma)]
        
        for beta in beta_values:
            for gamma in gamma_values:
                value = self.QKP_value_wrapper(beta, gamma, k, theta, bit_mapping)
                if value > self.best_value:
                    best_value = value

        return beta, gamma, best_value


    def parameter_optimization(self, k_range, theta_range, N_beta=50, N_gamma=50, bit_mapping='regular'):
        """Complete parameter optimization using grid search and BFGS."""
        best_value = -np.inf
        
        for k in k_range:
            for theta in theta_range:
                print(f"Parameters (k, θ): {int(k), theta}")
                
                # Grid search
                beta0, gamma0, value = self.grid_search(k, theta, N_beta, N_gamma, bit_mapping)

                # BFGS optimization
                def objective(params):
                    beta, gamma = params
                    return -self.QKP_value_wrapper(beta, gamma, k, theta, bit_mapping)
                
                result = minimize(objective, [beta0, gamma0], method='BFGS')

                if result.success and -result.fun > best_value:
                    print(f"New best solution (optim): {-result.fun}, {result.x[0]}, {result.x[1]}")
                    self.best_value > -result.fun
                    self.best_params = (result.x[0], result.x[1], k, theta)
