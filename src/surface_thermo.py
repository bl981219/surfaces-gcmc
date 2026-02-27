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

def Fe2O3_chemical_potential(T):
    NIST = np.array([
        [298.15, 87.400], [300, 87.402], [400, 91.691], [500, 100.284],
        [600, 110.415], [700, 121.020], [800, 131.653], [900, 142.124],
        [950, 147.271], [1000, 152.360], [1050, 157.323], [1100, 162.150],
        [1200, 171.388], [1300, 180.112], [1400, 188.372], [1500, 196.212],
        [1600, 203.671], [1700, 210.785], [1800, 217.583], [1900, 224.092],
        [2000, 230.335], [2100, 236.334], [2200, 242.105], [2300, 247.667],
        [2400, 253.034], [2500, 258.219]
    ])
    mu_IAP = -198.726593 / 6  # eV
    mu_ref = (-NIST[:, 1] * NIST[:, 0] + 15560) * 6.242 * 10 ** 18 / (6.0221408 * 10 ** 23)
    p = np.polyfit(np.concatenate([[0], NIST[:, 0]]), np.concatenate([[0], mu_ref]), 3)
    mu_0 = np.polyval(p, T)
    return mu_IAP + mu_0

def Fe_chemical_potential(T):
    NIST = np.array([
        [298.15,27.321], [300,27.321], [400,28.335], [500,30.323],
        [600,32.642], [700,35.063], [800,37.498], [900,39.922],
        [1000,42.342], [1042,43.372], [1100,44.815], [1184,46.865],
        [1200,47.257], [1300,49.612], [1400,51.822], [1500,53.907],
        [1600,55.884], [1665,57.117], [1700,57.776], [1800,59.610]
    ])
    mu_IAP = -16.939 / 2 # BCC Fe
    mu_ref = (-NIST[:, 1] * NIST[:, 0] + 4507) * 6.242 * 10 ** 18 / (6.0221408 * 10 ** 23)
    p = np.polyfit(np.concatenate([[0], NIST[:, 0]]), np.concatenate([[0], mu_ref]), 3)
    mu_0 = np.polyval(p, T)
    return mu_IAP + mu_0

def get_energy_and_atoms(folder_path, elements_order):
    """Robust parsing of energies and atom counts replacing os.popen grep."""
    try:
        outcar_path = os.path.join(folder_path, 'OUTCAR')
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'energy  without entropy=' in line or 'entropy=' in line:
                    energy = float(line.split('=')[-1].strip())
                    break
                    
        contcar_path = os.path.join(folder_path, 'CONTCAR')
        structure = Structure.from_file(contcar_path)
        
        # Count atoms matching the exact order defined in config.yaml
        comp_dict = structure.composition.get_el_amt_dict()
        atoms = [int(comp_dict.get(el, 0)) for el in elements_order]
        
        return energy, atoms
    except Exception as e:
        print(f"Error parsing {folder_path}: {e}")
        return None, None

def main(config_path, data_dir):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # 1. System Parameters
    T = config['thermodynamics']['temperature']
    mu_ref = config['thermodynamics']['m3gnet_o_ref']
    P_min, P_max = config['thermodynamics']['surface_pressure_range']
    elements = config['system']['elements']
    EABO3 = config['bulk_references']['ABO3']
    
    kb = 8.617333262145e-5 # eV/K
    F = 96485.3329 # C/mol
    R = 8.31446261815324 # J/(mol K)

    # 2. Setup Base Thermodynamic Math & Polynomial Fitting
    data = '''0, 2.986259541984733\n-1.084656084656082, 2.966793893129771\n-2.010582010582006, 2.9400763358778628\n-2.3280423280423292, 2.9278625954198474\n-2.6719576719576708, 2.9160305343511452\n-2.962962962962962, 2.904580152671756\n-3.2804232804232782, 2.8938931297709924\n-3.6243386243386198, 2.8820610687022903\n-3.9947089947089935, 2.870229007633588\n-4.365079365079367, 2.8591603053435115\n-4.788359788359788, 2.8477099236641226\n-5.317460317460316, 2.836259541984733\n-6.031746031746032, 2.8244274809160306\n-7.142857142857139, 2.8133587786259544'''
    lines = data.split('\n')
    pressure = np.log(10**np.array([float(line.split(',')[0]) for line in lines]))
    stoi = np.array([float(line.split(',')[1]) for line in lines])
    comp_ex = (3 - stoi) / 3 * 100
    
    energy_ls = 0.5 * (kb * T * pressure) + mu_ref
    z = np.polyfit(comp_ex, energy_ls, 1)
    p = np.poly1d(z)
    
    up_to_folder = 7
    comp = np.arange(1/81, (up_to_folder+1)/81, 1/81) * 100
    energy_ls_muO = p(comp)

    # Defect energy listings
    energy_ls_VLa = [-14.640747070299994, -14.444091796900011, -14.15240478520002, -13.624572753899997, -13.069213867099961, -12.387634277400025, -11.70123291020002][:up_to_folder]
    energy_ls_VFe = [-10.900695800800008, -10.63873291020002, -10.33978271479998, -9.686889648399983, -9.33514404289997, -8.36993408210003, -8.132141113300008][:up_to_folder]
    energy_ls_VSr = [-8.555603027400025, -8.536865234400011, -8.111511230500014, -7.761108398399983, -7.2529296875, -6.919433593800022, -6.528869629000042][:up_to_folder]

    p_muO_muLa = np.poly1d(np.polyfit(energy_ls_muO, energy_ls_VLa, 1))
    p_muO_muFe = np.poly1d(np.polyfit(energy_ls_muO, energy_ls_VFe, 1))
    p_muO_muSr = np.poly1d(np.polyfit(energy_ls_muO, energy_ls_VSr, 1))

    ref_VSr = p_muO_muSr(energy_ls_muO[0])
    ref_VLa = p_muO_muLa(energy_ls_muO[0])
    ref_VFe = p_muO_muFe(energy_ls_muO[0])

    mu_O_ref = mu_ref
    mu_O = np.arange(mu_O_ref - 3.5, mu_O_ref + 3.5, 0.001)

    comp_VM = [3.7037037, 7.40740741, 11.11111111, 14.81481481, 18.51851852, 22.22222222, 25.92592593]
    energy_ls_VFe_VM = [-10.96105957, -11.11697388, -11.3377889, -11.39762878, -11.51212158, -11.5945638, -11.72771345]
    energy_ls_VLa_VM = [-14.88830566, -15.07839966, -15.25860596, -15.44972229, -15.60380859, -15.75515747, -15.91832624]
    energy_ls_VSr_VM = [-8.78338623, -8.80593872, -8.92533366, -9.05870056, -9.17814941, -9.27721151, -9.35235596]

    tot_cation = 125
    p_VFe = np.poly1d(np.polyfit(comp_VM, energy_ls_VFe_VM, 1))
    p_VLa = np.poly1d(np.polyfit(comp_VM, energy_ls_VLa_VM, 1))
    p_VSr = np.poly1d(np.polyfit(comp_VM, energy_ls_VSr_VM, 1))

    p_VLa = p_VLa + (ref_VLa - p_VLa(0))
    p_VFe = p_VFe + (ref_VFe - p_VFe(0))
    p_VSr = p_VSr + (ref_VSr - p_VSr(0))

    # 3. Read Base Reference Structure
    base_path = Path(data_dir)
    energy_ref, atoms_ref = get_energy_and_atoms(base_path / 'ref', elements)
    if energy_ref is None:
        raise FileNotFoundError(f"Reference folder 'ref/' must exist in {data_dir} with OUTCAR and CONTCAR.")

    # 4. Setup Plotting
    fig, ax = plt.subplots()
    label_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 16}
    color_lst = [
        '#222222', '#E66D50', '#B0432B', '#7A1A06', '#F3A361', '#E7C66B', 
        '#D9A520', '#299D8F', '#257A7A', '#215766', '#297270', '#274753', 
        '#142C36', '#C9E265', '#8AB07C', '#4B7D50', '#1A4D2E'
    ]

    low_lst, high_lst = [], []
    energy_ref_1 = np.full_like(mu_O[mu_O <= energy_ls_muO[0]], 0)
    low_lst.append(energy_ref_1)
    energy_ref_2 = np.full_like(mu_O[mu_O >= energy_ls_muO[0]], 0)
    high_lst.append(energy_ref_2)

    ln_P_low = 2 * (mu_O[mu_O < energy_ls_muO[0]] - mu_O_ref) / (kb * T)
    log10_P_low = ln_P_low / np.log(10)
    ln_P_high = 2 * (mu_O[mu_O > energy_ls_muO[0]] - mu_O_ref) / (kb * T)
    log10_P_high = ln_P_high / np.log(10)

    ax.plot(log10_P_low, energy_ref_1, color=color_lst[0], label='Reference')
    ax.plot(log10_P_high, energy_ref_2, color=color_lst[0])

    folders = ['SrO2', 'SrO', 'La2O3', 'Fe', 'RP_GCMC']
    converged_folders = ['1', '2', '3']
    rp_subfolders = ['1', '2', '3_w_VO', '3_wo_VO']

    delta_Sr_delta_La = -0.0374909
    delta1 = (EABO3 - 81 * (mu_O[mu_O <= energy_ls_muO[0]]) - 11 * p_muO_muSr(mu_O[mu_O <= energy_ls_muO[0]]) - 27 * p_muO_muFe(mu_O[mu_O <= energy_ls_muO[0]]) - 16 * (p_muO_muLa(mu_O[mu_O <= energy_ls_muO[0]]) - delta_Sr_delta_La)) / 54
    delta2 = (EABO3 - 81 * (mu_O[mu_O >= energy_ls_muO[0]]) - 11 * ref_VSr - 27 * ref_VFe - 16 * (ref_VLa - delta_Sr_delta_La)) / 54
    delta2 = delta2 - (delta2[0] - delta1[-1])

    p_Sr1 = np.poly1d(np.polyfit(mu_O[mu_O <= energy_ls_muO[0]], p_muO_muSr(mu_O[mu_O <= energy_ls_muO[0]]), 1))
    p_La1 = np.poly1d(np.polyfit(mu_O[mu_O <= energy_ls_muO[0]], p_muO_muLa(mu_O[mu_O <= energy_ls_muO[0]]), 1))
    p_Fe1 = np.poly1d(np.polyfit(mu_O[mu_O <= energy_ls_muO[0]], p_muO_muFe(mu_O[mu_O <= energy_ls_muO[0]]), 1))

    # 5. Extract Folder Energies and Add to Plots
    for i, folder in enumerate(folders):
        print(f"Processing: {folder}")
        target_folders = rp_subfolders if folder == 'RP_GCMC' else converged_folders
        for j, subfolder in enumerate(target_folders):
            energy, atoms = get_energy_and_atoms(base_path / folder / subfolder, elements)
            
            if energy is not None:
                label = f"{folder} {j+1} layer" if folder != 'RP_GCMC' else f"Ruddlesden-Popper {subfolder}"
                stoi_diff = np.array(atoms) - np.array(atoms_ref)
                
                low = energy - energy_ref - stoi_diff[0] * (p_La1(mu_O[mu_O <= energy_ls_muO[0]]) + delta1 - delta_Sr_delta_La + p_VLa(stoi_diff[0]/tot_cation*100) - ref_VLa) - stoi_diff[1] * (p_Fe1(mu_O[mu_O <= energy_ls_muO[0]]) + delta1 + p_VFe(stoi_diff[1]/tot_cation*100) - ref_VFe) - stoi_diff[2] * mu_O[mu_O <= energy_ls_muO[0]] - stoi_diff[3] * (p_Sr1(mu_O[mu_O <= energy_ls_muO[0]]) + delta1 + p_VSr(stoi_diff[3]/tot_cation*100) - ref_VSr)
                ax.plot(log10_P_low, low, label=label, color=color_lst[3*i+j+1])
                
                high = energy - energy_ref - stoi_diff[0] * (ref_VLa + delta2 - delta_Sr_delta_La + p_VLa(stoi_diff[0]/tot_cation*100) - ref_VLa) - stoi_diff[1] * (ref_VFe + delta2 + p_VFe(stoi_diff[1]/tot_cation*100) - ref_VFe) - stoi_diff[2] * mu_O[mu_O >= energy_ls_muO[0]] - stoi_diff[3] * (ref_VSr + delta2 + p_VSr(stoi_diff[3]/tot_cation*100) - ref_VSr)
                ax.plot(log10_P_high, high, color=color_lst[3*i+j+1])

                low_lst.append(low)
                high_lst.append(high)

    # 6. Fill Lowest Energy Regions
    for k in range(len(mu_O[mu_O <= energy_ls_muO[0]]) - 1):
        ax.fill_between(log10_P_low[k:k+2], min([low_lst[m][k] for m in range(len(low_lst))]), y2=-200, facecolor=color_lst[np.argmin([low_lst[m][k] for m in range(len(low_lst))])], alpha=1)
    for k in range(len(mu_O[mu_O >= energy_ls_muO[0]])):
        ax.fill_between(log10_P_high[k:k+2], min([high_lst[m][k] for m in range(len(high_lst))]), y2=-200, facecolor=color_lst[np.argmin([high_lst[m][k] for m in range(len(high_lst))])], alpha=1)

    # Fill center color gap
    ax.fill_between([-2.5, 0.001], min([high_lst[m][0] for m in range(len(high_lst))]), y2=-200, facecolor=color_lst[np.argmin([high_lst[m][0] for m in range(len(high_lst))])], alpha=1)

    # 7. Axes formatting
    ax.axvline(x=np.log10(0.2), color='k', linestyle='--')
    ax.set_ylim(-80, 20)
    ax.set_xlim(P_min, P_max)
    ax.set_xlabel(r'$\log(P_{\mathrm{O_{2}}})\;(\mathrm{atm})$', fontdict=label_font)
    ax.set_ylabel(r'$\mathrm{Grand\;potentials\;(eV)}$', fontdict=label_font)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        
    ax.set_xticklabels(ax.get_xticks(), fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.set_yticklabels(ax.get_yticks(), fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))

    # Overpotential X-axis
    ax2 = ax.twiny()
    ita_min = (R * T / (4 * F)) * np.log(10**P_min / 0.2)
    ita_max = (R * T / (4 * F)) * np.log(10**P_max / 0.2)
    ax2.set_xticks(np.linspace(ita_min, ita_max, 9))
    ax2.set_xlabel(r'$\eta\;(\mathrm{V})$', fontdict=label_font)
    ax2.set_xlim(ita_min, ita_max)
    ax2.set_xticklabels(ax2.get_xticks(), fontweight='bold')
    ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    filename = f'surface_thermo_{T}K_generalized.png'
    fig.savefig(filename, bbox_inches='tight', dpi=600)
    print(f"Surface Phase diagram generated successfully: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Surface Phase Diagram")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing calculated structures')
    args = parser.parse_args()
    main(args.config, args.data_dir)