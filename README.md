# Surfaces-GCMC: Predicting Surface Atomic Structures on Doped Perovskite Oxides

This repository contains a multi-scale computational pipeline designed to predict surface reconstructions, terminations, and phase precipitates on complex doped perovskite oxides (e.g., La<sub>0.6</sub>Sr<sub>0.4</sub>FeO<sub>3-δ</sub>). It integrates Grand Canonical Monte Carlo (GCMC) sampling, Machine Learning Interatomic Potentials (ML-IAPs), and rigorous bulk-to-surface defect thermodynamics to generate highly accurate surface phase diagrams.

## Core Features
* **Accelerated GCMC Sampling:** Uses the M3GNet universal graph neural network to evaluate Monte Carlo structural moves (insertions, removals, exchanges, displacements) in-memory, bypassing expensive Density Functional Theory (DFT) steps for massive computational savings.
* **Hybrid VASP Verification:** Automatically triggers VASP at user-defined intervals to ground the ML-IAP energies and ensure high-fidelity structural relaxation.
* **Universal Surface Thermodynamics:** Generates multi-phase convex hull surface stability diagrams as a function of temperature ($T$), oxygen partial pressure ($P_{O_2}$), and overpotential ($\eta$). The mathematical model explicitly incorporates bulk defect chemistry, configurational mixing entropy (for fractional A/B-site doping), and self-limiting segregation principles.

## Prerequisites
* **Python 3.9+**
* `pymatgen`, `numpy`, `matplotlib`, `pyyaml`
* `m3gnet` (for ML-IAP evaluation)
* **VASP 5/6:** A licensed, compiled VASP executable accessible via your system's MPI run command.

## Repository Structure
```text
surfaces-gcmc/
├── src/
│   ├── bulk_thermo.py                 # Computes bulk phase diagrams
│   ├── competing_phases_identifier.py # Identifies competing phases for surface segregation
│   ├── gcmc_sampler.py                # ML-accelerated Monte Carlo sampling engine
│   └── surface_thermo.py              # Thermodynamic convex hull phase diagram generator
├── examples/
│   ├── example_config.yaml            # Template configuration file for testing
│   ├── test_POSCAR                    # Sample starting slab
│   └── vasp_data/                     # Example directory of relaxed VASP output data
├── config.yaml                        # Master configuration file for production runs
├── reproduce_figures.sh               # Script to reproduce manuscript figures
├── requirements.txt                   # Python environment dependencies
└── README.md
```

## How to Use the Pipeline

### Step 1: Prepare Bulk Defect Data (Prerequisite)
This model rigorously couples surface energies to the bulk reservoir. Before running the surface scripts on a new material, you must calculate the bulk defect formation energies using standard DFT. 
You will need to compute:
1. Cation vacancy formation energies as a function of oxygen vacancy concentration ($V_O$).
2. Cation vacancy formation energies as a function of metal vacancy concentration ($V_M$).

Input these polynomial arrays into the `cation_vacancy_vs_VO` and `cation_vacancy_vs_VM` blocks of your `config.yaml`.

### Step 2: Identify Competing Phases
Before running the surface sampling, you must evaluate the bulk stability against secondary phase precipitation. Run the bulk thermodynamics script to identify which phases compete with your parent material:
```bash
python src/competing_phases_identifier.py --config config.yaml
```
The identified competing phases (e.g., SrO, SrO<sub>2</sub>, Ruddlesden-Popper phases, or elemental metals) serve as the rational basis for seeding your initial surface structures in the next step.

### Step 3: Configure the GCMC Sampler
1. Construct an initial symmetric slab guided by the competing phases identified in Step 2, and save it as `POSCAR` in your working directory.
2. Ensure you have your VASP `INCAR`, `KPOINTS`, and `POTCAR` files in the same directory if you plan to use the hybrid VASP verification feature.
3. Edit `config.yaml` to define your strict Cartesian Z-boundaries for Monte Carlo moves (`region_mc_z`) and Grand Canonical insertions/removals (`region_gcmc_z`). 

### Step 4: Run the Sampler (HPC / Slurm Deployment)
Because GCMC requires thousands of evaluations, running this on a High-Performance Computing (HPC) cluster via a workload manager like Slurm is highly recommended. 

```bash
# Example command to run the sampler
python src/gcmc_sampler.py --config config.yaml --input POSCAR
```
*The sampler will output accepted geometries iteratively into the `output/trajectory/` directory.*

### Step 5: Extract Unique Phases
Analyze the generated `CONTCAR` files in the `output/trajectory/` folder. Cluster these geometries to identify the unique surface phases. 

Perform a final, tight VASP relaxation on these unique structures and place their `CONTCAR` and `OUTCAR` files into subdirectories within a target data folder (e.g., `data_dir/SrO2/1/`, `data_dir/RP_GCMC/1/`). Ensure you also include a `data_dir/ref/` folder containing your ideal, un-reconstructed reference slab.

### Step 6: Generate the Surface Phase Diagram
Once your unique phases are organized, execute the thermodynamics script to generate the convex hull:

```bash
python src/surface_thermo.py --config config.yaml --data_dir path/to/your/clustered/phases/
```
This script will automatically crawl your directory, apply the bulk defect calibrations and mixing entropies, and output a unified `.png` phase diagram mapping the lowest-energy surfaces.

## Authors & Citation
**Mengren Bill Liu, Hao Tang, Jing Yang, Xiaochen Du, Rafael Gómez-Bombarelli, and Bilge Yildiz**

If you use this code in your research, please cite our manuscript:
> Liu, M. B., Tang, H., Yang, J., Du, X., Gómez-Bombarelli, R., & Yildiz, B. "Predicting Surface Atomic Structures on Doped Perovskite Oxides Using Grand Canonical Monte Carlo: Model System of La<sub>0.6</sub>Sr<sub>0.4</sub>FeO<sub>3-δ</sub>" (Pending Publication).