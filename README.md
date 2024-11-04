# Variational-Quantum-Solver-Unit-Commitment
Variational Quantum ADMM Solver for Unit Commitment

This repository contains a hybrid quantum-classical implementation of the Alternating Direction Method of Multipliers (ADMM) algorithm using Variational Quantum Algorithms (VQA) to solve the Unit Commitment (UC) problem. The UC problem focuses on selecting the optimal subset of power units to meet demand while minimizing operational costs.

📋 Table of Contents

	•	Project Overview
	•	Problem Description
	•	Algorithm Overview
	•	Installation
	•	Usage
	•	Project Structure
	•	Results
	•	Contributing
	•	License

🚀 Project Overview

This project explores the use of the Quantum Approximate Optimization Algorithm (QAOA) within the ADMM framework to solve the Unit Commitment problem. By leveraging quantum computing (via IBM’s Qiskit platform) and classical optimization, we aim to address the complexity of this mixed-integer optimization problem.

🔍 Problem Description

The Unit Commitment (UC) problem involves:

	•	Selecting a subset of power generation units to be active (binary decision).
	•	Ensuring total power output matches demand, subject to operational constraints.
	•	Minimizing the total operational costs across a given time horizon.

Since UC is NP-hard, it becomes computationally intensive for large instances, making it a compelling use case for quantum-inspired algorithms.

🧑‍💻 Algorithm Overview

This hybrid quantum-classical approach leverages:

	1.	QAOA: Provides binary solutions for selecting which units are active.
	2.	ADMM Framework: Ensures that constraints are satisfied while optimizing continuous variables.
	3.	Hybrid Workflow: A classical optimizer tunes the QAOA parameters (γ, β), and the quantum circuit generates feasible binary solutions.

⚙️ Installation

To run the code locally, follow these steps:

	1.	Clone the Repository

git clone https://github.com/your_username/variational_quantum_solver_UC.git  
cd variational_quantum_solver_UC


	2.	Create a Virtual Environment

python -m venv venv  
source venv/bin/activate  # On Windows: venv\Scripts\activate


	3.	Install Dependencies

pip install -r requirements.txt


	4.	Set up Qiskit Account (Optional for IBM Quantum)
If using IBM Quantum resources, create an account here and save your API token:

from qiskit import IBMQ  
IBMQ.save_account('your_token_here')  



▶️ Usage

	1.	Prepare the Input Data:
Define your input matrix with power units and demand requirements.
	2.	Run the Algorithm:

python main.py


	3.	Adjust Parameters:
Modify QAOA or ADMM settings in the config.yaml file.
	4.	Visualize Results:

python plot_results.py


Project Structure
UC

	├── figures				 # various figures of results
	│
	├── logs
	│   └── admm_optimizer.log               # logs file of the ADMM optimizer
	│
	├── results
	│   └── admm_results.pkl                 # ADMM result files
	│
	├── scripts
	│   ├── experiments                      # Main directory for experiment scripts
	│   │   ├── admm_tutorial.py             # Tutorial script for ADMM
	│   │   ├── fake_hardware_exemple.py     # Simulation on fake quantum hardware
	│   │   ├── post_process_simu.py         # Post-processing simulation results
	│   │   ├── run_admm_cross-terms.py      # ADMM with cross-term constraints
	│   │   ├── run_admm_hardware.py         # ADMM on real quantum hardware
	│   │   ├── run_admm_simulation.py       # ADMM simulation script
	│   │   ├── run_qaoa_multi-vars.py       # Multi-variable QAOA experiment
	│   │   ├── run_rho_convergence.py       # Analyze rho convergence behavior
	│   │   └── run_transpiler_settings.py   # Evaluate various transpiler settings
	│   │
	│   ├── solvers                          # Directory for UC solver implementations
	│   │   ├── admm_optimizer.py            # ADMM optimizer implementation
	│   │   ├── classical_solver_UC.py       # Classical UC solver for benchmarking
	│   │   ├── temp.py                      # Temp script for prototyping
	│   │    temp2.py                        # Another temp script for development
	│   │
	│   └── utils                            # Utilities and supporting functions
	│       ├── models.py                    # Model definitions for UC
	│       └── utils.py                     # Helper functions for data handling, logging
	│
	└── requirements.txt                     # Python dependencies for project setup


📊 Results

The project produces:

	•	Binary Solutions: Indicating which units are active.
	•	Optimized Power Outputs: Continuous values for active units.
	•	Plots: Visualizing the evolution of QAOA parameters (γ, β) over iterations.

🤝 Contributing

We welcome contributions!

	1.	Fork the repository.
	2.	Create a new branch:

git checkout -b feature-branch  


	3.	Commit your changes:

git commit -m "Describe your changes"  


	4.	Push to your fork and submit a pull request.

📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

📧 Contact

For questions or feedback, please contact Julien-Pierre or open an issue in this repository.

This README.md provides a comprehensive overview of the project, ensuring users understand the problem, the algorithm, and how to set up and use the code effectively.
