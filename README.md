# KEM

This repository provides python codes for the KEM kernel-based expectation-maximization (KEM) algorithm accompanying the following paper.


**🚩 File Tree**
```
.
├── KEM_experiment
│   ├── KEM_EXPE.py
│   ├── experiment_auxiliary.py
│   ├── experiment_case_illustration.ipynb
│   └── transfer_pi_to_dicom.ipynb
├── KEM_simulation
│   ├── KEM_SIMU.py
│   ├── data_generate.py
│   ├── optimal bwd csv
│   │   └── random100_all.csv
│   ├── simulation_auxiliary.py
│   ├── simulation_case_show.ipynb
│   └── simulation_study.ipynb
├── README.md
└── utils.py
```

## Prerequisites

**🚩 Python Packages**

- `pydicom`
- `SimpleITK`
- `tensorflow`
- `multiprocessing`
- `glob`
- `gc`
- `sklearn`