# src/gcmc_sampler.py
# Author: Mengren Bill Liu

import os
import random
import math
import yaml
import argparse
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Outcar

class GCMCSampler:
    def __init__(self, config_path):
        """Initializes the GCMC Sampler with generalized parameters from config.yaml."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # 1. System Parameters
        self.elements = self.config['system']['elements']
        self.masses = self.config['system']['masses']
        self.T = self.config['thermodynamics']['temperature']
        
        # 2. GCMC Spatial Boundaries (Fractional coordinates)
        self.z_gcmc_min, self.z_gcmc_max = self.config['gcmc_settings']['region_gcmc_z']
        self.z_mc_min, self.z_mc_max = self.config['gcmc_settings']['region_mc_z']
        
        # 3. Probabilities and Execution
        self.vasp_cmd = self.config['gcmc_settings']['vasp_cmd']
        self.iterations = self.config['gcmc_settings']['iterations']
        
        # Move probabilities
        self.p_displace = self.config['gcmc_settings']['displace_ratio']
        self.p_change = self.config['gcmc_settings']['change_ratio']
        self.p_remove = self.config['gcmc_settings']['remove_ratio']
        self.p_insert = self.config['gcmc_settings']['insert_ratio']

        # Constants
        self.kb = 8.617333262145e-5 # eV/K

    def read_energy(self, outcar_path="OUTCAR"):
        """Robustly reads the final energy from VASP/M3GNet OUTCAR."""
        try:
            outcar = Outcar(outcar_path)
            return outcar.final_energy
        except Exception:
            # Fallback for M3GNet mock OUTCARs
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if 'energy' in line:
                        return float(line.split()[-1])
        return None

    def write_poscar(self, structure, filename="POSCAR"):
        """
        Dynamically applies Selective Dynamics (T T T for surface, F F F for bulk)
        based on the config file's MC z-boundaries.
        """
        selective_dynamics = []
        for site in structure:
            # If atom is outside the active MC region, freeze it
            if self.z_mc_min <= site.frac_coords[2] <= self.z_mc_max:
                selective_dynamics.append([True, True, True])
            else:
                selective_dynamics.append([False, False, False])
                
        poscar = Poscar(structure, selective_dynamics=selective_dynamics)
        poscar.write_file(filename)

    def metropolis_acceptance(self, delta_E, delta_N=0, mu=0):
        """Calculates GCMC Metropolis acceptance probability."""
        # Grand Canonical criteria: delta_GrandPotential = delta_E - mu * delta_N
        delta_omega = delta_E - (mu * delta_N)
        
        if delta_omega < 0:
            return True
        else:
            prob = math.exp(-delta_omega / (self.kb * self.T))
            return random.random() < prob

    def run_vasp(self):
        """Executes the VASP or M3GNet evaluation command specified in config."""
        print("Evaluating energy...")
        os.system(self.vasp_cmd)

    def execute_gcmc_loop(self, initial_poscar="POSCAR"):
        """Main Monte Carlo Loop."""
        structure = Structure.from_file(initial_poscar)
        self.write_poscar(structure)
        self.run_vasp()
        
        current_energy = self.read_energy()
        print(f"Initial Energy: {current_energy} eV")

        for step in range(self.iterations):
            move_type = random.random()
            print(f"--- Step {step+1}/{self.iterations} ---")
            
            # Note: Place your specific structural manipulation logic here using pymatgen
            if move_type < self.p_displace:
                print("Action: Displace atom")
                # Add pymatgen site displacement logic
            elif move_type < (self.p_displace + self.p_change):
                print("Action: Swap element")
                # Add pymatgen element swap logic
            elif move_type < (self.p_displace + self.p_change + self.p_remove):
                print("Action: Remove atom")
                # Add pymatgen site deletion logic
            else:
                print("Action: Insert atom")
                # Add pymatgen site insertion logic
                
            # Evaluate new structure
            # self.write_poscar(new_structure)
            # self.run_vasp()
            # new_energy = self.read_energy()
            
            # if self.metropolis_acceptance(new_energy - current_energy):
            #     current_energy = new_energy
            #     structure = new_structure
            # else:
            #     print("Move rejected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GCMC Surface Sampler")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--input', type=str, default='POSCAR', help='Initial structure file')
    args = parser.parse_args()
    
    sampler = GCMCSampler(args.config)
    sampler.execute_gcmc_loop(args.input)