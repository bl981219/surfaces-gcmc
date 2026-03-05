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
    except Exception:
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Generate Surface Phase Diagram")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_dir', type=str, default='.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    data_dir = args.data_dir

    T = config['thermodynamics']['temperature']
    mu_ref = config['thermodynamics']['m3gnet_o_ref']
    P_min, P_max = config['thermodynamics']['surface_pressure_range']
    
    elements = config['system']['elements']
    stoich_list = config['system']['target_stoichiometry']
    stoich = dict(zip(elements, stoich_list))
    
    oxygen_el = 'O'
    cations = [el for el in elements if el != oxygen_el]
    
    kb = 8.617333262145e-5; F = 96485.3329; R = 8.31446261815324
    EABO3 = config['bulk_references']['ABO3']['energy']
    
    vo_cfg = config['cation_vacancy_vs_VO']
    vm_cfg = config['cation_vacancy_vs_VM']
    mixing_offsets = vo_cfg.get('mixing_entropy_offsets', {})
    
    tot_cation_bulk = sum(stoich[c] for c in cations) 
    tot_cation_slab = vm_cfg['tot_cation']
    
    # 1. Defect Polynomial Fitting (Oxygen Vacancy Dependent)
    pressure_data = np.log(10**np.array(vo_cfg['pressure_data_log10']))
    comp_ex = (3 - np.array(vo_cfg['stoi_data'])) / 3 * 100
    energy_ls = 0.5 * (kb * T * pressure_data) + mu_ref
    p_muO = np.poly1d(np.polyfit(comp_ex, energy_ls, 1))
    
    num_pts = len(vo_cfg[f'energy_ls_V{cations[0]}'])
    comp = np.arange(1/stoich[oxygen_el], (num_pts+1)/stoich[oxygen_el], 1/stoich[oxygen_el]) * 100
    energy_ls_muO = p_muO(comp)

    p_muO_mu = {}
    ref_V = {}
    for c in cations:
        z = np.polyfit(energy_ls_muO, vo_cfg[f'energy_ls_V{c}'], 1)
        p_muO_mu[c] = np.poly1d(z)
        ref_V[c] = p_muO_mu[c](energy_ls_muO[0])

    # 2. Defect Polynomial Fitting (Metal Vacancy Dependent)
    comp_VM = vm_cfg['comp_VM']
    p_VM = {}
    for c in cations:
        p = np.poly1d(np.polyfit(comp_VM, vm_cfg[f'energy_ls_V{c}'], 1))
        p_VM[c] = p + (ref_V[c] - p(0)) 

    # 3. Chemical Potential Arrays
    mu_O_ref = mu_ref
    mu_O = np.arange(mu_O_ref - 3.5, mu_O_ref + 3.5, 0.001)
    mu_O_low = mu_O[mu_O <= energy_ls_muO[0]]
    mu_O_high = mu_O[mu_O >= energy_ls_muO[0]]

    # 4. Bulk Calibration Constraints
    sum_cations_low = sum(stoich[c] * (p_muO_mu[c](mu_O_low) + mixing_offsets.get(c, 0.0)) for c in cations)
    sum_cations_high = sum(stoich[c] * (ref_V[c] + mixing_offsets.get(c, 0.0)) for c in cations)
    
    delta1 = (EABO3 - stoich[oxygen_el] * mu_O_low - sum_cations_low) / tot_cation_bulk
    delta2 = (EABO3 - stoich[oxygen_el] * mu_O_high - sum_cations_high) / tot_cation_bulk
    delta2 = delta2 - (delta2[0] - delta1[-1])

    p_c1 = {c: np.poly1d(np.polyfit(mu_O_low, p_muO_mu[c](mu_O_low), 1)) for c in cations}

    ln_P_low = 2*(mu_O_low - mu_O_ref)/(kb*T)
    log10_P_low = ln_P_low/np.log(10)
    ln_P_high = 2*(mu_O_high - mu_O_ref)/(kb*T)
    log10_P_high = ln_P_high/np.log(10)

    # 5. Extract Reference Data
    base_path = Path(data_dir)
    energy_ref, atoms_ref_list = get_energy_and_atoms(base_path / 'ref', elements)
    if energy_ref is None:
        raise FileNotFoundError(f"Missing reference data in {base_path / 'ref'}")
    atoms_ref = dict(zip(elements, atoms_ref_list))

    # ==========================================================
    # 6. PRE-SCAN: Count all valid structures to generate colors
    # ==========================================================
    plot_settings = config.get('plot_settings', {})
    folders = plot_settings.get('folder_order', sorted(os.listdir(base_path)))
    
    valid_paths = []
    for f_name in folders:
        f_path = base_path / f_name
        if not f_path.is_dir() or f_name == 'ref': 
            continue
        for t_name in sorted(os.listdir(f_path)):
            path = f_path / t_name
            if path.is_dir():
                # Verify it has a valid energy before counting it
                e, _ = get_energy_and_atoms(path, elements)
                if e is not None:
                    valid_paths.append((f_name, t_name, path))

    # Mathematically slice the "turbo" colormap to give exactly the right number of distinct colors
    total_lines = len(valid_paths) + 1 # +1 for the reference line
    color_palette = plt.cm.turbo(np.linspace(0.05, 0.95, total_lines))
    # ==========================================================

    # Plot Setup
    fig, ax = plt.subplots()
    label_font = {'family':'sans-serif', 'weight': 'bold', 'size': 16}
    
    low_lst = [np.full_like(mu_O_low, 0)]
    high_lst = [np.full_like(mu_O_high, 0)]
    dynamic_color_lst = []
    
    # Plot Reference structure using the first color in our new dynamic palette
    c_ref = color_palette[0]
    ax.plot(log10_P_low, low_lst[0], label='Reference', color=c_ref)
    ax.plot(log10_P_high, high_lst[0], color=c_ref)
    dynamic_color_lst.append(c_ref)

    # 7. Universal Directory Plotting
    for idx, (f_name, t_name, path) in enumerate(valid_paths):
        energy, atoms_list = get_energy_and_atoms(path, elements)
        atoms = dict(zip(elements, atoms_list))
        stoi_diff = {el: atoms[el] - atoms_ref[el] for el in elements}
        
        low = energy - energy_ref - stoi_diff[oxygen_el] * mu_O_low
        high = energy - energy_ref - stoi_diff[oxygen_el] * mu_O_high
        
        for c in cations:
            offset = mixing_offsets.get(c, 0.0)
            term_low = p_c1[c](mu_O_low) + delta1 + offset + p_VM[c](stoi_diff[c]/tot_cation_slab*100) - ref_V[c]
            low -= stoi_diff[c] * term_low
            
            term_high = ref_V[c] + delta2 + offset + p_VM[c](stoi_diff[c]/tot_cation_slab*100) - ref_V[c]
            high -= stoi_diff[c] * term_high

        label = f"{f_name} {t_name}"
        
        # Grab the exact, unique color pre-calculated for this specific phase
        c_hex = color_palette[idx + 1]
        
        ax.plot(log10_P_low, low, label=label, color=c_hex)
        ax.plot(log10_P_high, high, color=c_hex)
        
        low_lst.append(low)
        high_lst.append(high)
        dynamic_color_lst.append(c_hex)

    # 8. Generate the Phase Envelope (Convex Hull) 
    for i in range(len(mu_O_low)-1):
        min_idx = np.argmin([ls[i] for ls in low_lst])
        ax.fill_between(log10_P_low[i:i+2], min([ls[i] for ls in low_lst]), y2=-200, 
                        facecolor=dynamic_color_lst[min_idx], alpha=1)
                        
    for i in range(len(mu_O_high)-1):
        min_idx = np.argmin([ls[i] for ls in high_lst])
        ax.fill_between(log10_P_high[i:i+2], min([ls[i] for ls in high_lst]), y2=-200, 
                        facecolor=dynamic_color_lst[min_idx], alpha=1)
    
    min_idx_transition = np.argmin([ls[0] for ls in high_lst])
    ax.fill_between([log10_P_low[-1], log10_P_high[0]], min([ls[0] for ls in high_lst]), y2=-200, 
                    facecolor=dynamic_color_lst[min_idx_transition], alpha=1)

    # 9. Render and Format
    ax.axvline(x=np.log10(0.2), color='k', linestyle='--')
    
    ax.legend(frameon=True, loc="center left", bbox_to_anchor=(1.05, 0.5), handlelength=1.4, framealpha=0.92, fontsize="x-small")
    
    ax.set_ylim(-80, 20)
    ax.set_xlim(P_min, P_max)
    ax.set_xlabel(r'$\log(P_{\mathrm{O_{2}}})\;(\mathrm{atm})$', fontdict=label_font)
    ax.set_ylabel(r'$\mathrm{Grand\;potentials\;(eV)}$', fontdict=label_font)
    
    for spine in ax.spines.values(): spine.set_linewidth(2)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    
    ax2 = ax.twiny()
    ita_min = (R*T/(4*F))*np.log(10**P_min/0.2)
    ita_max = (R*T/(4*F))*np.log(10**P_max/0.2)
    ax2.set_xticks(np.linspace(ita_min, ita_max, 9))
    ax2.set_xlim(ita_min, ita_max)
    ax2.set_xlabel(r'$\eta\;(\mathrm{V})$', fontdict=label_font)
    ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.setp(ax2.get_xticklabels(), fontweight='bold')

    os.makedirs('output', exist_ok=True)
    filename = f'output/surface_thermo_{T}K.png'
    fig.savefig(filename, bbox_inches='tight', dpi=600)
    print(f"Surface phase diagram successfully generated: {filename}")

if __name__ == "__main__":
    main()