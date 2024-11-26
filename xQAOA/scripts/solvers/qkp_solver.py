import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from itertools import product


from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_ibm_runtime import SamplerV2


class QKPOptimizer:
    def __init__(self, v, w, c, mixer, p=1, run_hardware=False, backend=None,
                 sampler=None, pass_manager=None, optimal_solution=None,
                 generate_jobs=False, speedup_computation=True):
        self.v = v
        self.w = w
        self.c = c
        self.mixer = mixer
        self.backend = backend
        self.run_hardware = run_hardware
        self.pass_manager = pass_manager
        self.optimal_solution = optimal_solution
        self.generate_jobs = generate_jobs
        self.speedup_computations = speedup_computation
        self.p = p

        self.n = len(v)
        self.best_bitstring = None
        self.best_value = -np.inf
        self.best_weight = -np.inf
        if sampler == None:
            self.sampler = StatevectorSampler()
        else:
            self.sampler = sampler
        if generate_jobs:
            self.list_transpile_qc = []
        self.dict_all_parameters = {}


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


    def QKP(self, betas, gammas, k, theta=None, bit_mapping='regular',
            run_single_job=False, shots=5000):
        
        p = self.logistic_bias(k)
        qc = QuantumCircuit(self.n)

        # Initial state preparation
        for i in range(self.n):
            angle = 2 * np.arcsin(np.sqrt(p[i]))
            qc.ry(angle, i)
        qc.barrier()

        for i in range(self.p):

            # Cost unitary
            self.apply_cost_unitary(qc, gammas[i])
            qc.barrier()

            # Mixer application
            if self.mixer == 'X':
                self.apply_X_mixer(qc, betas[i])

            elif self.mixer == 'hourglass':
                self.apply_hourglass_mixer(qc, betas[i], p)

            elif self.mixer == 'copula':
                p2 = self.logistic_bias(k)
                self.apply_copula_mixer(qc, betas[i], p, p2, theta)

        qc.measure_all()


        # Start the timer
        start_time = time.time()

        # Run on real hardware
        if self.run_hardware:
            # Transpilation
            isa_qc = self.pass_manager.run(qc)
            backend = self.backend
            # print(isa_qc)

            if self.generate_jobs == False or run_single_job:
                job = self.sampler.run([isa_qc], shots=shots)
                result = job.result()[0]
            else:
                self.list_transpile_qc.append(isa_qc)
                return None, None, None, None, None

        # Run on local simulator
        else:
            job = self.sampler.run([qc], shots=shots)
            result = job.result()[0]

        # Stop the timer
        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"Quantum circuit execution time: {execution_time:.4f} seconds")

        counts = result.data.meas.get_counts()

        # Find best solution
        best_solution = max(counts, key=counts.get)

        if bit_mapping == 'regular':
            best_value = sum(int(best_solution[i]) * self.v[i] for i in range(self.n))
            total_weight = sum(int(best_solution[i]) * self.w[i] for i in range(self.n))

        
        if bit_mapping == 'inverse':
            best_value = sum((1 - int(best_solution[i])) * self.v[i] for i in range(self.n))
            total_weight = sum((1 - int(best_solution[i])) * self.w[i] for i in range(self.n))


        # Save parameters results of each runs
        self.dict_all_parameters[f"{betas},{gammas}"] = best_value

        # Make sure the solution is valid
        if total_weight <= self.c:
            return best_solution, float(best_value), total_weight, counts, True
        else:
            return best_solution, float(best_value), total_weight, counts, False
        


    def QKP_value_wrapper(self, betas, gammas, k, theta, bit_mapping, shots):
        """Wrapper that tracks the best bitstring while returning only the value."""
        bitstring, value, weight, counts, valid = self.QKP(betas, gammas, k, theta, bit_mapping, shots=shots)

        if self.generate_jobs == False:
            if (value > self.best_value) and (valid==True):
                print(f"New best solution: {int(value)} -- [{betas}, {gammas}]")
                self.best_bitstring = bitstring
                self.best_value = value
                self.best_weight = weight
                self.best_params = (betas, gammas)
            return value
        
    from tqdm import tqdm
    import numpy as np

    def grid_search(self, k, theta, N_beta, N_gamma, bit_mapping, shots, show_progress=True):
        """
        Grid search for optimization of β and γ, with optional progress bar.

        Args:
            k (int): The value of k for the search.
            theta (float): The value of theta for the search.
            N_beta (int): Number of grid points for beta.
            N_gamma (int): Number of grid points for gamma.
            bit_mapping (str): Bit mapping strategy.
            shots (int): Number of shots for simulation.
            show_progress (bool): Whether to display a progress bar.

        Returns:
            tuple: The best beta, gamma, and their corresponding value.
        """
        best_value = -np.inf
        found_opt_sol = False

        beta_values = [np.pi * i / N_beta for i in range(N_beta)]
        gamma_values = [2 * np.pi * j / N_gamma for j in range(N_gamma)]

        # Use tqdm for progress tracking if show_progress is enabled
        beta_iterator = tqdm(product(beta_values, repeat=self.p), desc="Grid Search β", disable=not show_progress)

        for betas in beta_iterator:
            for gammas_combo in product(gamma_values, repeat=self.p):
                if not found_opt_sol:
                    # print('(BETAS, GAMMAS)', betas, gammas_combo)
                    value = self.QKP_value_wrapper(betas, gammas_combo,
                                                   k, theta,
                                                   bit_mapping, shots=shots)

                    if value > best_value:
                        best_value = value
                        best_beta, best_gamma = betas, gammas_combo

                    # Early stopping if optimal solution is found
                    if (self.best_value == self.optimal_solution and 
                        self.speedup_computations):
                        print("Found optimal solution")
                        found_opt_sol = True
                        break

        return best_beta, best_gamma, best_value


    # def grid_search(self, k, theta, N_beta, N_gamma, bit_mapping, shots):
    #     """Grid search for optimization of β and γ."""
    #     best_value = -np.inf
    #     found_opt_sol = False

    #     beta_values = [np.pi * i / N_beta for i in range(N_beta)]
    #     gamma_values = [2 * np.pi * j / N_gamma for j in range(N_gamma)]

    #     # beta_values = np.linspace(np.pi/4-0.4, np.pi/4+0.4)
        
    #     for beta in beta_values:
    #         for gamma in gamma_values:
    #             # print(beta, gamma)
    
    #             if found_opt_sol == False:
    #                 # print("found opt sol:", found_opt_sol)
    #                 value = self.QKP_value_wrapper(beta, gamma, k, theta, bit_mapping, shots=shots)

    #                 if self.best_value == self.optimal_solution and self.speedup_computations:
    #                     print("Found optimal solution")
    #                     found_opt_sol = True


        return beta, gamma, best_value


    def parameter_optimization(self, k_range, theta_range, N_beta=50, N_gamma=50,
                               bit_mapping='regular', shots=5000):
        """Complete parameter optimization using grid search and BFGS."""
        best_value = -np.inf
        
        for k in k_range:
            for theta in theta_range:
                print(f"Parameters (k, θ): {int(k), theta}")
                
                # Grid search
                beta0, gamma0, value = self.grid_search(k, theta, N_beta, N_gamma, bit_mapping, shots=shots)



    
    def generate_circuits(self, k_range, theta_range, N_beta=50, N_gamma=50):

        beta_values = [np.pi * i / N_beta for i in range(N_beta)]
        gamma_values = [2 * np.pi * j / N_gamma for j in range(N_gamma)]
        list_qc = []

        for k in k_range:
            for theta in theta_range:

                for beta in range(N_beta):
                    for gamma in range(N_gamma):
                                
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
                        list_qc.append(qc)

        return list_qc
    

    def transpile_circuits(self, list_qc, pass_manager, show_progess_bar=True):
        """ Transpile a list of quantum circuits."""
        list_isa_qc = []
        iterator = tqdm(list_qc, desc="Transpiling circuits") if show_progess_bar else list_qc

        for qc in iterator:
            isa_qc = pass_manager.run(qc)
            list_isa_qc.append(isa_qc)

        return list_isa_qc


