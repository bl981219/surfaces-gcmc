# src/competing_phases_identifier.py
# Author: Mengren Bill Liu

import os
import yaml
import argparse
import numpy as np
import json

def main(config_path, data_path):
    # 1. Load configuration dynamically
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    T = config['thermodynamics']['temperature']
    mu_O_ref = config['thermodynamics']['m3gnet_o_ref']
    
    # Dynamically extract bounds from the bulk_pressure_range in config.yaml
    refs = config['bulk_references']

    F = 96485.3329
    R = 8.31446261815324
    kb = 8.617333262145e-5

    # 2. Setup output file
    os.makedirs('output', exist_ok=True)
    output_file_path = os.path.join('output', 'competing_phases_result.txt')

    # 3. Read the JSON trajectory from data/
    try:
        with open(data_path, 'r') as f:
            mu_data = json.load(f)['data']
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Ensure you saved the arrays as JSON.")
        return

    mu_La = mu_data.get('La', [])
    mu_Fe = mu_data.get('Fe', [])
    mu_O = mu_data.get('O', [])
    mu_Sr = mu_data.get('Sr', [])
    
    num_points = len(mu_La)

    # 4. Evaluate and Write to File
    with open(output_file_path, 'w') as out_file:
        for i in range(num_points):
            # Build current step dictionary from the arrays
            mu_dict = {
                'La': mu_La[i],
                'Fe': mu_Fe[i],
                'O': mu_O[i],
                'Sr': mu_Sr[i]
            }
            
            ln_P = 2 * (mu_dict['O'] - mu_O_ref) / (kb * T)
            log10_P = ln_P / np.log(10)

            eta = (R * T / (4 * F)) * np.log(np.exp(ln_P) / 0.2)
            
            # Print to terminal
            print(f"log10(PO2): {log10_P:.4f}")
            print(f"eta: {eta:.4f}")
            
            # Write to text file
            out_file.write(f"log10(PO2): {log10_P:.4f}\n")
            out_file.write(f"eta: {eta:.4f}\n")
            
            # Evaluate every phase mapped in config.yaml
            for phase_name, phase_data in refs.items():
                if phase_name == 'ABO3':
                    continue
                    
                stoich = phase_data['stoich']
                E_ref = phase_data['energy']
                
                # Generalized dot product: sum(stoich_i * mu_i)
                energy_mu = sum(stoich.get(el, 0) * mu_dict.get(el, 0) for el in set(stoich))
                
                # If sum(mu) > E_ref, phase is stable
                if energy_mu > E_ref:
                    print(f"{phase_name} stable: True")
                    out_file.write(f"{phase_name} stable: True\n")
                    
            print('-------------------')
            out_file.write('-------------------\n')

    print(f"Success! Output saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate phase stability from chemical potential trajectories.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    # Point the default data argument directly to the new JSON file
    parser.add_argument('--input', type=str, default='examples/mu_trajectory.json', help='Path to JSON data file')
    args = parser.parse_args()
    main(args.config, args.input)