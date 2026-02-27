# Surfaces-GCMC: Surface Thermodynamics & Reconstruction Modeler

A computational workflow for predicting the surface reconstructions, point defect concentrations, and phase stability of complex oxides under operando conditions (temperature and gas pressure). 

This package uses a hybrid approach combining Machine Learning Interatomic Potentials (M3GNet) and Density Functional Theory (DFT) to accelerate Grand Canonical Monte Carlo (GCMC) sampling of slab models.

## Overview
This repository contains three main modules:
1. **Bulk Thermodynamics (`bulk_thermo.py`)**: Uses linear programming to calculate the bulk phase diagram of complex oxides (e.g., LaSrFeO3) against competing secondary phases as a function of temperature and $P_{\mathrm{O_2}}$.
2. **GCMC Surface Sampling (`gcmc_sampler.py`)**: Performs Grand Canonical Monte Carlo sampling of surface atoms and vacancies. It uses M3GNet as a rapid surrogate model to pre-relax and screen configurations before validating ground-state structures with VASP.
3. **Surface Phase Diagram (`surface_thermo.py`)**: Analyzes the converged GCMC trajectories and bulk reference energies to construct the final surface phase diagram (Grand Potential vs. oxygen chemical potential/overpotential).

## Installation

It is highly recommended to run this code within an isolated Conda environment.

```bash
# Clone the repository
git clone [https://github.com/bl981219/surfaces-gcmc.git](https://github.com/bl981219/surfaces-gcmc.git)
cd surfaces-gcmc

# Create and activate the environment
conda create -n surfaces_gcmc python=3.9 -y
conda activate surfaces_gcmc

# Install dependencies
pip install -r requirements.txt