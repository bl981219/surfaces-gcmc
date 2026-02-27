# src/bulk_thermo.py
# Author: Mengren Bill Liu

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import linprog

def oxygen_chemical_potential(T, P, overbinding):
    """Calculates oxygen chemical potential using NIST data and DFT corrections."""
    nist_data = np.array([
        [298.15, 205.147], [400, 206.308], [600, 211.044], 
        [800, 216.126], [1000, 220.875], [1200, 225.209], 
        [1400, 229.158], [1600, 232.768]
    ])
    mu_dft = -9.79981 + 2 * overbinding
    mu_ref = (-nist_data[:, 1] * nist_data[:, 0] + 8683) * 6.242e18 / 6.022e23
    p = np.polyfit([0, *nist_data[:, 0]], [0, *mu_ref], 3)
    mu_0 = np.polyval(p, T)
    mu_p = 8.617e-5 * T * np.log(P)
    return 0.5 * (mu_dft + mu_p) # Simplified as per original logic

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    T = config['thermodynamics']['temperature']
    overbinding = config['thermodynamics']['o2_overbinding_correction']
    P_min, P_max = config['thermodynamics']['bulk_pressure_range']
    P_array = np.linspace(P_min, P_max, 100)
    
    refs = config['bulk_references']
    E_ABO3 = refs['ABO3']
    
    # Setup plot
    fig, ax = plt.subplots()
    color_lst = ['#ffe881', '#f4e6d3', '#9a9a9f', 'blue', 'red', 'white']
    x_lst = []

    for P in P_array:
        mu_O = oxygen_chemical_potential(T, 10**P, overbinding)
        
        # c array matches: [La2O3, Fe, FeO, Fe2O3, SrO, SrO2, SFO, LFO, O]
        c = np.array([refs['La2O3'], refs['Fe'], refs['FeO'], refs['Fe2O3'], 
                      refs['SrO'], refs['SrO2'], refs['SFO'], refs['LFO'], mu_O])
        
        A_eq = np.array([
            [2,0,0,0,0,0,0,27,0],    # La 
            [0,1,1,2,0,0,27,27,0],   # Fe 
            [3,0,1,3,1,2,81,81,1],   # O 
            [0,0,0,0,1,1,27,0,0]     # Sr 
        ])
        b_eq = config['system']['target_stoichiometry']
        
        bounds = [(0, None)] * 8 + [(None, None)] # Last bound for mu_O
        
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        if res.success:
            if E_ABO3 - res.fun < 0:
                ax.plot(T, P, '.', color=color_lst[-1], alpha=0.5)
            else:
                diff = [np.linalg.norm(res.x - x) for x in x_lst]
                if not diff or all(d > 1e-5 for d in diff):
                    x_lst.append(res.x)
                    ax.plot(T, P, '.', color=color_lst[len(x_lst)-1], alpha=0.5)
                else:
                    index = diff.index(min(diff))
                    ax.plot(T, P, '.', color=color_lst[index], alpha=0.5)

    label_font = {'weight': 'bold', 'size': 16}
    ax.set_xlabel('Temperature (K)', fontdict=label_font)
    ax.set_ylabel(r'$\log(P_{\mathrm{O_2}})\;(\mathrm{atm})$', fontdict=label_font)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.savefig('bulk_phase_diagram_generalized.png', bbox_inches='tight', dpi=150)
    print("Phase diagram saved to bulk_phase_diagram_generalized.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Bulk Phase Diagram")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)