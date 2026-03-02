# src/competing_phases_identifier.py
# Author: Mengren Bill Liu

import os
import yaml
import argparse
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description="Evaluate phase stability from chemical potential trajectories.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--input', type=str, default='examples/mu_trajectory.json', help='Path to JSON data file')
    args = parser.parse_args()

    # 1. Load configuration dynamically
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    T = config['thermodynamics']['temperature']
    mu_O_ref = config['thermodynamics']['m3gnet_o_ref']
    refs = config['bulk_references']

    F = 96485.3329
    R = 8.31446261815324
    kb = 8.617333262145e-5

    # 2. Setup output file
    os.makedirs('output', exist_ok=True)
    output_file_path = os.path.join('output', 'competing_phases_result.txt')

    # 3. Read the JSON trajectory dynamically
    try:
        with open(args.input, 'r') as f:
            # Handles both {"data": {...}} wrapper and direct {...} dicts
            json_content = json.load(f)
            mu_data = json_content.get('data', json_content) 
    except FileNotFoundError:
        print(f"Error: {args.input} not found. Ensure you saved the arrays as JSON.")
        return

    # Dynamically extract elements and array lengths from the JSON
    elements = list(mu_data.keys())
    if not elements:
        print("Error: No data found in JSON.")
        return
    num_points = len(mu_data[elements[0]])

    # 4. Evaluate and Write to File
    with open(output_file_path, 'w') as out_file:
        for i in range(num_points):
            # Dynamically build the mu_dict for the current step
            mu_dict = {el: mu_data[el][i] for el in elements if i < len(mu_data[el])}
            
            if 'O' not in mu_dict:
                print("CRITICAL: Oxygen ('O') chemical potential is missing from the trajectory data.")
                return
            
            ln_P = 2 * (mu_dict['O'] - mu_O_ref) / (kb * T)
            log10_P = ln_P / np.log(10)
            eta = (R * T / (4 * F)) * np.log(np.exp(ln_P) / 0.2)
            
            # Print and write conditions
            header = f"log10(PO2): {log10_P:.4f} | eta: {eta:.4f}"
            print(header)
            out_file.write(header + "\n")
            
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
                    msg = f"  -> {phase_name} stable: True"
                    print(msg)
                    out_file.write(msg + "\n")
                    
            print('-------------------')
            out_file.write('-------------------\n')

    print(f"\nSuccess! Output saved to {output_file_path}")

if __name__ == "__main__":
    main()