# src/stability_checker.py
# Author: Mengren Bill Liu

import yaml
import argparse
import numpy as np
import csv

def main(config_path, data_path):
    # 1. Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    T = config['thermodynamics']['temperature']
    mu_O_ref = config['thermodynamics']['m3gnet_o_ref']
    refs = config['bulk_references']

    F = 96485.3329
    R = 8.31446261815324
    kb = 8.617333262145e-5

    # 2. Read the chemical potentials from CSV
    try:
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            data_rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please provide a CSV with element columns (e.g., La, Fe, O, Sr).")
        return

    # 3. Evaluate stability conditions
    for row in data_rows:
        # Convert CSV strings to floats
        mu_dict = {el: float(val) for el, val in row.items()}
        
        # We must have Oxygen to calculate PO2
        if 'O' not in mu_dict:
            continue
            
        mu_O_val = mu_dict['O']
        ln_P = 2 * (mu_O_val - mu_O_ref) / (kb * T)
        log10_P = ln_P / np.log(10)

        # Only evaluate inside your specified pressure bounds
        if -20 < log10_P < 10:
            eta = (R * T / (4 * F)) * np.log(np.exp(ln_P) / 0.2)
            print(f"log10(PO2): {log10_P:.4f}")
            print(f"eta: {eta:.4f}")
            
            # Loop through every reference phase in config.yaml
            for phase_name, phase_data in refs.items():
                if phase_name == 'ABO3':  # Skip the target LSF phase
                    continue
                    
                stoich = phase_data['stoich']
                E_ref = phase_data['energy']
                
                # DYNAMIC DOT PRODUCT: 
                # Automatically multiplies mu and stoich based on matching element names.
                # If an element is missing, .get() defaults to 0.
                current_energy = sum(stoich.get(el, 0) * mu_dict.get(el, 0) for el in set(stoich) | set(mu_dict))
                
                if current_energy > E_ref:
                    print(f"{phase_name} stable: True")
                    
            print('-------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate phase stability from chemical potential trajectories.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, default='data/mu_trajectory.csv', help='Path to CSV data file')
    args = parser.parse_args()
    main(args.config, args.data)