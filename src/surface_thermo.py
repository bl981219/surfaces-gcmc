# src/surface_thermo.py
# Author: Mengren Bill Liu

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from pymatgen.core import Structure

def get_energy_and_atoms(folder_path, elements_order):
    """Robust parsing of energies and atom counts."""
    try:
        outcar_path = os.path.join(folder_path, 'OUTCAR')
        with open(outcar_path, 'r') as f:
            for line in reversed(f.readlines()):
                if 'energy' in line:
                    energy = float(line.split()[-1])
                    break
        structure = Structure.from_file(os.path.join(folder_path, 'CONTCAR'))
        comp = structure.composition.get_el_amt_dict()
        return energy, [int(comp.get(el, 0)) for el in elements_order]
    except Exception as e:
        print(f"Skipping {folder_path}: {e}")
        return None, None

def main(config_path, data_dir):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # 1. System Parameters Dynamically Loaded
    T = config['thermodynamics']['temperature']
    mu_ref = config['thermodynamics']['m3gnet_o_ref']
    P_min, P_max = config['thermodynamics']['surface_pressure_range']
    elements = config['system']['elements']
    stoich = config['system']['target_stoichiometry'] # Dynamically loads [16, 27, 81, 11]
    tot_cation = sum(stoich) - stoich[2] # Sum of cations (16+27+11 = 54)
    EABO3 = config['bulk_references']['ABO3']
    
    kb = 8.617e-5; F = 96485.3; R = 8.314

    # 2. Defect Polynomial Fitting (Retained from manuscript math)
    pressure_data = np.log(10**np.array([0, -1.08, -2.01, -2.32, -2.67, -2.96, -3.28, -3.62, -3.99, -4.36, -4.78, -5.31, -6.03, -7.14]))
    stoi_data = np.array([2.986, 2.966, 2.940, 2.927, 2.916, 2.904, 2.893, 2.882, 2.870, 2.859, 2.847, 2.836, 2.824, 2.813])
    comp_ex = (3 - stoi_data) / 3 * 100
    
    energy_ls = 0.5 * (kb * T * pressure_data) + mu_ref
    p_muO = np.poly1d(np.polyfit(comp_ex, energy_ls, 1))
    
    comp = np.arange(1/stoich[2], 8/stoich[2], 1/stoich[2]) * 100
    energy_ls_muO = p_muO(comp)

    # Note: Explicit defect energies kept from your specific manuscript data fits
    energy_ls_VLa = [-14.64, -14.44, -14.15, -13.62, -13.06, -12.38, -11.70]
    energy_ls_VFe = [-10.90, -10.63, -10.33, -9.68, -9.33, -8.36, -8.13]
    energy_ls_VSr = [-8.55, -8.53, -8.11, -7.76, -7.25, -6.91, -6.52]

    p_muO_muLa = np.poly1d(np.polyfit(energy_ls_muO, energy_ls_VLa, 1))
    p_muO_muFe = np.poly1d(np.polyfit(energy_ls_muO, energy_ls_VFe, 1))
    p_muO_muSr = np.poly1d(np.polyfit(energy_ls_muO, energy_ls_VSr, 1))

    ref_VSr, ref_VLa, ref_VFe = p_muO_muSr(energy_ls_muO[0]), p_muO_muLa(energy_ls_muO[0]), p_muO_muFe(energy_ls_muO[0])

    mu_O = np.arange(mu_ref - 3.5, mu_ref + 3.5, 0.001)

    # 3. Read Base Reference Structure
    base_path = Path(data_dir)
    energy_ref, atoms_ref = get_energy_and_atoms(base_path / 'ref', elements)
    if energy_ref is None:
        raise FileNotFoundError(f"Reference folder 'ref/' must exist in {data_dir} with OUTCAR and CONTCAR.")

    # 4. Generalized Plotting Math
    fig, ax = plt.subplots()
    color_lst = ['#222222', '#E66D50', '#B0432B', '#7A1A06', '#F3A361', '#E7C66B', '#D9A520']
    
    delta_Sr_delta_La = -0.0374909
    
    # DYNAMIC STOICHIOMETRY APPLIED HERE
    mu_O_low = mu_O[mu_O <= energy_ls_muO[0]]
    mu_O_high = mu_O[mu_O >= energy_ls_muO[0]]
    
    # EABO3 - (O*muO) - (Sr*muSr) - (Fe*muFe) - (La*muLa) / Total Cations
    delta1 = (EABO3 - stoich[2]*mu_O_low - stoich[3]*p_muO_muSr(mu_O_low) - stoich[1]*p_muO_muFe(mu_O_low) - stoich[0]*(p_muO_muLa(mu_O_low) - delta_Sr_delta_La)) / tot_cation
    delta2 = (EABO3 - stoich[2]*mu_O_high - stoich[3]*ref_VSr - stoich[1]*ref_VFe - stoich[0]*(ref_VLa - delta_Sr_delta_La)) / tot_cation
    delta2 = delta2 - (delta2[0] - delta1[-1])

    # Plot formatting...
    ax.set_xlim(P_min, P_max)
    ax.set_ylim(-80, 20)
    ax.set_xlabel(r'$\log(P_{\mathrm{O_{2}}})\;(\mathrm{atm})$', fontweight='bold')
    ax.set_ylabel(r'$\mathrm{Grand\;potentials\;(eV)}$', fontweight='bold')
    
    ax2 = ax.twiny()
    ita_min = (R * T / (4 * F)) * np.log(10**P_min / 0.2)
    ita_max = (R * T / (4 * F)) * np.log(10**P_max / 0.2)
    ax2.set_xlim(ita_min, ita_max)
    ax2.set_xlabel(r'$\eta\;(\mathrm{V})$', fontweight='bold')

    filename = f'surface_thermo_{T}K_generalized.png'
    fig.savefig(filename, bbox_inches='tight', dpi=600)
    print(f"Surface Phase diagram generated successfully: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_dir', type=str, default='.')
    args = parser.parse_args()
    main(args.config, args.data_dir)