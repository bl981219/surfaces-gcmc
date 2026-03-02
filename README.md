# Surfaces-GCMC: Predicting Oxides Surface Atomic Structures

This repository contains a multi-scale computational pipeline designed to predict surface reconstructions, terminations, and phase precipitates on complex doped perovskite oxides (e.g., La<sub>0.6</sub>Sr<sub>0.4</sub>FeO<sub>3-δ</sub>). It integrates Grand Canonical Monte Carlo (GCMC) sampling, Machine Learning Interatomic Potentials (ML-IAPs), and rigorous bulk-to-surface defect thermodynamics to generate highly accurate surface phase diagrams.

## Core Features
* **Bulk Thermodynamics:** Calculates bulk decomposition phase diagrams and evaluates competing phases under specific environmental conditions to predict potential surface segregates.
* **Accelerated GCMC Sampling:** Uses the M3GNet universal graph neural network to evaluate Monte Carlo structural moves (insertions, removals, exchanges, displacements) in-memory, bypassing expensive Density Functional Theory (DFT) steps for massive computational savings.
* **Hybrid VASP Verification:** Automatically triggers VASP at user-defined intervals to ground the ML-IAP energies and ensure high-fidelity structural relaxation.
* **Universal Surface Thermodynamics:** Generates multi-phase convex hull surface stability diagrams as a function of temperature ($T$), oxygen partial pressure ($P_{O_2}$), and overpotential ($\eta$). The mathematical model explicitly incorporates bulk defect chemistry, configurational mixing entropy (for fractional cation site doping), and self-limiting segregation principles.

## Prerequisites
* **Python 3.9** (Recommended to maintain strict compatibility with the `m3gnet` backend and its TensorFlow dependencies).
* **VASP 5/6:** A licensed, compiled VASP executable accessible via your system's MPI run command (required for hybrid DFT verification).

## Installation
Because this pipeline integrates machine learning libraries and computational chemistry tools, we **highly recommend** installing it inside an isolated virtual environment (e.g., Conda) to avoid dependency conflicts.

```bash
# 1. Create and activate a Conda environment
conda create -n surfaces_gcmc python=3.9 -y
conda activate surfaces_gcmc

# 2. Clone the repository
git clone https://github.com/bl981219/surfaces-gcmc.git
cd surfaces-gcmc

# 3. Install the package (Editable mode recommended for researchers)
pip install -e .
```

## Quick Start (Reproducing Manuscript Figures)
To verify your installation and reproduce the figures from the manuscript using the provided La<sub>0.6</sub>Sr<sub>0.4</sub>FeO<sub>3-δ</sub> (LSF) data, simply run the automated bash script:

```bash
bash reproduce_figures.sh
```
*Check the `output/` directory for the generated PNG phase diagrams and text logs.*

---

## Step-by-Step Pipeline Usage

If you want to run the pipeline manually or adapt it for your own material, follow these steps. (The examples below use the provided 1073K test files).

### Step 1: Prepare Bulk Defect Data (Prerequisite)
This model rigorously couples surface energies to the bulk reservoir. Before running the surface scripts on a new material, you must calculate the bulk defect formation energies using standard DFT. Input these polynomial arrays into the `cation_vacancy_vs_VO` and `cation_vacancy_vs_VM` blocks of your configuration file.

### Step 2: Evaluate Bulk Thermodynamics & Competing Phases
Before surface sampling, evaluate the bulk stability to identify which secondary phases might precipitate at the surface. 

First, calculate the bulk decomposition phase diagram:
```bash
surfaces-bulk-thermo --config examples/example_config_1073K.yaml
```

Next, read your chemical potential trajectories to find the specific phases that compete with your parent material:
```bash
surfaces-competing-phases --config examples/example_config_1073K.yaml --input examples/mu_trajectory.json
```
*The identified competing phases serve as the rational basis for seeding your initial surface structures in the next step.*

### Step 3: Run the GCMC Sampler (HPC Recommended)
Construct an initial symmetric slab guided by the competing phases identified in Step 2. Because GCMC requires thousands of evaluations, running this on a High-Performance Computing (HPC) cluster via a workload manager like Slurm is highly recommended. 

```bash
surfaces-gcmc-sampler --config examples/example_config_1073K.yaml --input examples/test_POSCAR
```
*The sampler will output accepted geometries iteratively into the `output/trajectory/` directory.*

### Step 4: Extract Unique Phases & Relax
Analyze the generated `CONTCAR` files in the `output/trajectory/` folder. Cluster these geometries to identify the unique surface phases. Perform a final, tight VASP relaxation on these unique structures and place their `CONTCAR` and `OUTCAR` files into subdirectories within a target data folder (e.g., `examples/converged_structures/SrO2/1/`). Ensure you also include a `ref/` folder containing your ideal, un-reconstructed reference slab.

### Step 5: Generate the Surface Phase Diagram
Once your unique phases are organized, execute the thermodynamics script to generate the convex hull:

```bash
surfaces-gcmc-thermo --config examples/example_config_1073K.yaml --data_dir ./examples/converged_structures/
```
This script will automatically crawl your directory, apply the bulk defect calibrations and mixing entropies, and output a unified `.png` phase diagram mapping the lowest-energy surfaces.

## Citation
If you use this code in your research, please cite our manuscript:
> Liu, M. B., Tang, H., Yang, J., Du, X., Gómez-Bombarelli, R., & Yildiz, B. "Predicting Surface Atomic Structures on Doped Perovskite Oxides Using Grand Canonical Monte Carlo: Model System of La<sub>0.6</sub>Sr<sub>0.4</sub>FeO<sub>3-δ</sub>" (Pending Publication).