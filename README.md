# Variational-Quantum-Solver-Unit-Commitment
Variational Quantum Solver for Unit Commitment

This repository contains the implementation of a Variational Quantum Algorithms (VQA) to solve the Unit Commitment (UC) problem. The UC problem focuses on selecting the optimal subset of power units to meet demand while minimizing operational costs.

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

This project explores the use of the Quantum Approximate Optimization Algorithm (QAOA) to solve the Unit Commitment problem. By leveraging quantum computing (via IBM’s Qiskit platform) and classical optimization, we aim to address the complexity of this mixed-integer optimization problem.

🔍 Problem Description

The Unit Commitment (UC) problem involves:

	•	Selecting a subset of power generation units to be active (binary decision).
	•	Ensuring total power output matches demand, subject to operational constraints.
	•	Minimizing the total operational costs across a given time horizon.

Since UC is NP-hard, it becomes computationally intensive for large instances, making it a compelling use case for quantum-inspired algorithms.

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
