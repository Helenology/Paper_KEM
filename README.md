# KEM

This repository provides python codes for the KEM kernel-based expectation-maximization (KEM) algorithm accompanying the paper [A Semiparametric Gaussian Mixture Model for Chest CT-based 3D Blood Vessel Reconstruction](https://doi.org/10.1093/biostatistics/kxae013) by Qianhan Zeng, Jing Zhou, Ying Ji, and Hansheng Wang (Biostatistics, 2024).

## Citation
```bibtex
@article{Zeng_biostatistics_2024,
    author = {Zeng, Qianhan and Zhou, Jing and Ji, Ying and Wang, Hansheng},
    title = "{A semiparametric Gaussian mixture model for chest CT-based 3D blood vessel reconstruction}",
    journal = {Biostatistics},
    pages = {kxae013},
    year = {2024},
    month = {04},
    issn = {1465-4644},
    doi = {10.1093/biostatistics/kxae013},
    url = {https://doi.org/10.1093/biostatistics/kxae013},
    eprint = {https://academic.oup.com/biostatistics/advance-article-pdf/doi/10.1093/biostatistics/kxae013/58914368/kxae013.pdf},
}
```

## Pre-requisites (Python Packages)

- `pydicom`
- `SimpleITK`
- `tensorflow`
- `multiprocessing`
- `glob`
- `gc`
- `sklearn`


## File Tree

Below are the contents contained in this repo.

```
Paper_KEM
├── KEM_experiment
│   ├── 3D Reconstruction
│   ├── KEM_EXPE.py
│   ├── __pycache__
│   ├── experiment_SPE-2023-11-08.csv
│   ├── experiment_auxiliary.py
│   ├── 【LIDC】case_illustration(KEM_GT).ipynb
│   ├── 【LIDC】case_illustration.ipynb
│   └── 【LUNA0】SPE_experiment.ipynb
├── KEM_simulation
│   ├── KEM_SIMU.py
│   ├── Plot_Metrics_Compare.R
│   ├── __pycache__
│   ├── data_generate.py
│   ├── 【All0】simulation_study.ipynb
│   ├── 【Figure1】simulation_case_show.ipynb
│   ├── 【compare】simulation_case_show.ipynb
│   └── 【results】
├── Models
│   ├── GMM.py
│   ├── Kmeans.py
│   └── __pycache__
├── README.md
├── __pycache__
│   └── utils.cpython-39.pyc
└── utils.py

10 directories, 17 files
```



## Codes Overview

The repository consists of two parts: part 1 is the code for simulation studies, and part 2 is that for real data experiment.

### PART 1. [Simulation](/KEM_simulation/)

#### (1). Codes

- The following code files can be used to reproduce simulation results presented in the main paper.


|File name| Description |
|-------------|---------------|
|**`data_generate.py`**| Python codes to generate parameters and synthetic responses in simulation. |
|**`KEM_SIMU.py`**| Python codes to implement the KEM algorithm in simulation. |
|**`【Figure1】simulation_case_show.ipynb`**| Jupyter notebook to plot Figure 1. |
|**`【All0】simulation_study.ipynb`**| Jupyter notebook to conduct the bandwidth selection and comparison between the KEM method, the $k$-means method and the GMM method. Results in Figure 2 are produced here.|
|**`Plot_Metrics_Compare.R`**| R codes to boxplot Figure 2. |
|**`【compare】simulation_case_show.ipynb`**| Jupyter notebook to visualize an example comparison between the KEM method, the $k$-means method and the GMM method. Only serve as an intuitive example.|

#### (2). Results

- `【results】`: the folder of the results in Figure 2


### PART 2. [Experiment](/KEM_experiment/)

#### (1). Codes

- The following code files can be used to reproduce experiment results presented in the main paper.

|File name| Description |
|-------------|---------------|
|**`experiment_auxiliary.py`**| Python codes for the auxiliary functions in the experiment. |
|**`KEM_EXPE.py`**| Python to implement the KEM algorithm tailored for the experiment, which is a little different from that of the simulation. |
|**`【LIDC】case_illustration.ipynb`**| Jupyter notebook to produce Figure 3.|
|**`【LIDC】case_illustration(KEM_GT).ipynb`**| Jupyter notebook to produce Figure D2 in the Supplementary Materials.|
|**`【LUNA0】SPE_experiment.ipynb`**| Jupyter notebook to produce the results in Figure 4. |


> 1. Note that the dicom files should be load into RadiAnt (for example) for 3D visualization.
> 2. The colormaps used in this article are `plt.cm.bone_r` and `plt.cm.bone`.
> 3. I tried to directly use the `KEM_SIMU.py` in the real data experiment. However, sometimes the results were not satisfying. I found that it is better to use $\hat\mu^{(t)}$ in updating $\hat\sigma^{(t+1)}$ in the M step, rather than to use $\hat\mu^{(t+1)}$. This should be the only difference.


### (2). Results

- `experiment_SPE-2023-11-08.csv`: the csv file of the results in Figure 4
- `3D Reconstruction`: the folder of the 3D reconstructed dicom files

