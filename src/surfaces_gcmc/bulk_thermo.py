# src/bulk_thermo.py
# Author: Mengren Bill Liu

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import linprog
import os

def oxygen_chemical_potential(T, P, mu_0_o2):
    """
    Calculates the oxygen atom chemical potential for a given temperature (K) and pressure (atm).
    Uses experimental DFT binding energy.
    """
    nist_data = np.array([
        [100, 231.094], [200, 207.823], [250, 205.630], [298.15, 205.147],
        [300, 205.148], [350, 205.506], [400, 206.308], [450, 207.350],
        [500, 208.524], [600, 211.044], [700, 213.611], [800, 216.126],
        [900, 218.552], [1000, 220.875], [1100, 223.093], [1200, 225.209],
        [1300, 227.229], [1400, 229.158], [1500, 231.002], [1600, 232.768],
        [1700, 234.462], [1800, 236.089], [1900, 237.653], [2000, 239.160],
    ])

    mu_ref = (-nist_data[:, 1] * nist_data[:, 0] + 8683) * 6.242e18 / 6.0221408e23
    p = np.polyfit([0, *nist_data[:, 0]], [0, *mu_ref], 3)
    mu_0 = np.polyval(p, T)
    mu_p = 8.617e-5 * T * np.log(P)

    return 0.5 * (mu_0_o2 + mu_0 + mu_p)

def main():
    parser = argparse.ArgumentParser(description="Calculate Bulk Phase Diagram")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # 1. Load Parameters Dynamically
    T_min, T_max = config['thermodynamics']['temperature_range']
    P_min, P_max = config['thermodynamics']['bulk_pressure_range']
    mu_0_o2 = config['thermodynamics']['mu_0_o2']
    
    T_array = np.linspace(T_min, T_max, 100)
    P_array = np.linspace(P_min, P_max, 100)
    
    F = 96485.3329
    R = 8.31446261815324

    refs = config['bulk_references']
    E_ABO3 = refs['ABO3']['energy']
    target_stoich = refs['ABO3']['stoich']
    
    # 2. Dynamically Construct Linear Programming Matrices
    # Automatically pulls elements based on ABO3 defs (e.g., ['La', 'Fe', 'O', 'Sr'])
    active_elements = list(target_stoich.keys()) 
    b_eq = np.array([target_stoich[el] for el in active_elements])
    
    phases = [p for p in refs if p != 'ABO3']
    phases.append('O_gas')
    
    A_eq = np.zeros((len(active_elements), len(phases)))
    c = np.zeros(len(phases))
    
    for j, phase in enumerate(phases):
        if phase == 'O_gas':
            if 'O' in active_elements:
                A_eq[active_elements.index('O'), j] = 1.0
        else:
            c[j] = refs[phase]['energy']
            for i, el in enumerate(active_elements):
                A_eq[i, j] = refs[phase]['stoich'].get(el, 0.0)

    bounds = [(0, None)] * (len(phases) - 1) + [(None, None)]

    # 3. Setup Plot
    fig, ax = plt.subplots()
    label_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 16}
    color_lst = ["#a1a9d0", '#f0988c', "#b883d3", '#cfeaf1', 'white']
    
    x_lst = []
    T_boundary = []
    P_boundary = []
    last_phase = -1

    # 4. Thermodynamic Iteration
    for T in T_array:
        for P in P_array:
            testO = oxygen_chemical_potential(T, 10**P, mu_0_o2)
            c[-1] = testO 
            
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                if E_ABO3 - res.fun < 0:
                    # Target ABO3 phase is stable
                    ax.plot(T, P, '.', color=color_lst[-1], alpha=0.5)
                    ita = (R*T/(4*F)) * np.log(10**P/0.2)
                    
                    if last_phase != -1:
                        T_boundary.append(T)
                        P_boundary.append(P)
                        last_phase = -1
                else:
                    diff = [np.linalg.norm(res.x - x) for x in x_lst]
                    # Identify new phase decomposition regions
                    if all(d > 1e-5 for d in diff) or len(x_lst) == 0:
                        x_lst.append(res.x)
                        color_idx = (len(x_lst) - 1) % (len(color_lst) - 1) # Loop structural colors
                        ax.plot(T, P, '.', color=color_lst[color_idx], alpha=0.5)
                        T_boundary.append(T)
                        P_boundary.append(P)
                        last_phase = len(x_lst)
                        
                        products = [phases[i] for i, x in enumerate(res.x) if x > 1e-6]
                        print(f"\nNew Phase Region Identified at T={T:.0f}K, log10(P_O2)={P:.2f}:")
                        print(f"LSF decomposes into → {', '.join(products)}")
                    else:
                        index = diff.index(min(diff))
                        color_idx = index % (len(color_lst) - 1)
                        ax.plot(T, P, '.', color=color_lst[color_idx], alpha=0.5)
                        if last_phase != index:
                            T_boundary.append(T)
                            P_boundary.append(P)
                            last_phase = index

    # 5. Format and Save Axes
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    # Safely make the formatted tick labels bold
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.set_xlabel(r'$\mathrm{Temperature (K)}$', fontdict=label_font)
    ax.set_ylabel(r'$\log(P_{\mathrm{O_{2}}})\;(\mathrm{atm})$', fontdict=label_font)

    # CREATE OUTPUT DIRECTORY AND SAVE
    os.makedirs('output', exist_ok=True) # Safely creates the folder if it doesn't exist
    save_path = os.path.join('output', 'bulk_phase_diagram.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\nPhase diagram saved to {save_path}")

if __name__ == "__main__":
    main()