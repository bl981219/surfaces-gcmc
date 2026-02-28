# src/gcmc_sampler.py
# Author: Mengren Bill Liu

import os
import copy
import random
import math
import yaml
import argparse
import numpy as np
import shutil
import csv
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Outcar

# Silence TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pymatgen")

try:
    from m3gnet.models import Relaxer
    M3GNET_AVAILABLE = True
except ImportError:
    M3GNET_AVAILABLE = False
    print("Warning: m3gnet is not installed.")

class GCMCSampler:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.T = self.config['thermodynamics']['temperature']
        self.mu_dict = self.config['thermodynamics']['chemical_potentials']
        
        self.gcmc_species = self.config['gcmc_settings'].get('gcmc_species', ['O'])
        self.min_dist = self.config['gcmc_settings'].get('min_distance', 0.5)
        
        self.z_gcmc_min, self.z_gcmc_max = self.config['gcmc_settings']['region_gcmc_z']
        self.z_mc_min, self.z_mc_max = self.config['gcmc_settings']['region_mc_z']
        
        self.vasp_cmd = self.config['gcmc_settings'].get('vasp_cmd', 'echo "Missing vasp cmd"')
        freq_val = self.config['gcmc_settings'].get('vasp_verification_freq', 'inf')
        self.vasp_freq = float('inf') if str(freq_val).lower() == 'inf' else int(freq_val)
        
        self.iterations = self.config['gcmc_settings']['iterations']
        
        self.p_displace = self.config['gcmc_settings']['displace_ratio']
        self.p_change = self.config['gcmc_settings']['change_ratio']
        self.p_remove = self.config['gcmc_settings']['remove_ratio']
        self.kb = 8.617333262145e-5

        # Directory and File Paths
        self.work_dir = "output/working"
        self.traj_dir = "output/trajectory"
        self.poscar_path = os.path.join(self.work_dir, "POSCAR")
        self.contcar_path = os.path.join(self.work_dir, "CONTCAR")
        self.outcar_path = os.path.join(self.work_dir, "OUTCAR")
        self.log_file = "output/gcmc_log.csv"

        if M3GNET_AVAILABLE:
            print("Loading M3GNet Relaxer...")
            self.relaxer = Relaxer(relax_cell=False)
        else:
            self.relaxer = None

    def read_energy(self):
        try:
            return Outcar(self.outcar_path).final_energy
        except Exception:
            if os.path.exists(self.outcar_path):
                with open(self.outcar_path, 'r') as f:
                    for line in reversed(f.readlines()):
                        if 'energy' in line:
                            return float(line.split()[-1])
        return None

    def write_poscar(self, structure):
        selective_dynamics = [[self.z_mc_min <= site.coords[2] <= self.z_mc_max]*3 for site in structure]
        Poscar(structure, selective_dynamics=selective_dynamics).write_file(self.poscar_path)

    def get_active_indices(self, structure, z_min, z_max, species=None):
        indices = []
        for i, site in enumerate(structure):
            if z_min <= site.coords[2] <= z_max:
                if species is None or site.species_string in species:
                    indices.append(i)
        return indices

    def metropolis_acceptance(self, delta_E, delta_N_dict):
        delta_omega = delta_E
        for el, dN in delta_N_dict.items():
            delta_omega -= self.mu_dict.get(el, 0) * dN
            
        if delta_omega < 0:
            return True
        return random.random() < math.exp(-delta_omega / (self.kb * self.T))

    def attempt_move(self, structure, move_type):
        new_struct = copy.deepcopy(structure)
        if "selective_dynamics" in new_struct.site_properties:
            new_struct.remove_site_property("selective_dynamics")
            
        delta_N = {}
        valid_move = False
        
        if move_type == 'displace':
            active_idx = self.get_active_indices(new_struct, self.z_mc_min, self.z_mc_max)
            if active_idx:
                idx = random.choice(active_idx)
                for _ in range(10): 
                    new_x = random.uniform(0, structure.lattice.a)
                    new_y = random.uniform(0, structure.lattice.b)
                    new_z = random.uniform(self.z_mc_min, self.z_mc_max)
                    
                    new_struct.replace(idx, new_struct[idx].species, coords=[new_x, new_y, new_z], coords_are_cartesian=True)
                    
                    distances = new_struct.distance_matrix[idx]
                    distances[idx] = float('inf')
                    if np.min(distances) > self.min_dist:
                        valid_move = True
                        break

        elif move_type == 'change':
            active_idx = self.get_active_indices(new_struct, self.z_mc_min, self.z_mc_max)
            if len(active_idx) >= 2:
                idx1, idx2 = random.sample(active_idx, 2)
                sp1, sp2 = new_struct[idx1].species, new_struct[idx2].species
                if sp1 != sp2:
                    new_struct.replace(idx1, sp2)
                    new_struct.replace(idx2, sp1)
                    valid_move = True

        elif move_type == 'remove':
            active_idx = self.get_active_indices(new_struct, self.z_gcmc_min, self.z_gcmc_max, self.gcmc_species)
            if active_idx:
                idx = random.choice(active_idx)
                el = new_struct[idx].species_string
                new_struct.remove_sites([idx])
                delta_N[el] = -1
                valid_move = True

        elif move_type == 'insert':
            insert_el = random.choice(self.gcmc_species)
            for _ in range(20):
                new_x = random.uniform(0, structure.lattice.a)
                new_y = random.uniform(0, structure.lattice.b)
                new_z = random.uniform(self.z_gcmc_min, self.z_gcmc_max)
                
                new_struct.insert(0, insert_el, [new_x, new_y, new_z], coords_are_cartesian=True)
                
                distances = new_struct.distance_matrix[0]
                distances[0] = float('inf')
                if np.min(distances) > self.min_dist:
                    delta_N[insert_el] = 1
                    valid_move = True
                    break
                else:
                    new_struct.remove_sites([0]) 
                    
        return new_struct, delta_N, valid_move

    def run_m3gnet(self):
        if not self.relaxer:
            raise RuntimeError("M3GNet relaxer is not loaded.")
        structure_poscar = Poscar.from_file(self.poscar_path).structure
        relax_results = self.relaxer.relax(structure_poscar, verbose=False, steps=4000)
        
        final_structure = relax_results['final_structure']
        Poscar(final_structure).write_file(self.contcar_path)
        
        final_energy = float(relax_results['trajectory'].energies[-1])
        with open(self.outcar_path, "w") as output_file:
            output_file.write(f'  energy  without entropy=    {final_energy}\n')
        return final_energy

    def evaluate_energy(self, step):
        if self.vasp_freq != float('inf') and step > 0 and step % self.vasp_freq == 0:
            print(f"[{step}] Executing VASP Verification...")
            # Navigate into the working directory, run VASP, then navigate back
            os.system(f"cd {self.work_dir} && {self.vasp_cmd}")
        else:
            print(f"[{step}] Executing M3GNet Evaluation...")
            self.run_m3gnet()

    def execute_gcmc_loop(self, initial_poscar="POSCAR"):
        # Setup directories
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Move_Type', 'Valid_Move', 'Accepted', 'Delta_E', 'Current_Energy'])

        structure = Structure.from_file(initial_poscar)
        self.write_poscar(structure)
        
        self.evaluate_energy(step=0)
        current_energy = self.read_energy()
        structure = Structure.from_file(self.contcar_path)
        print(f"Initial Energy: {current_energy} eV\n")
        
        shutil.copy(self.contcar_path, os.path.join(self.traj_dir, "step_0_CONTCAR"))

        for step in range(1, self.iterations + 1):
            r = random.random()
            if r < self.p_displace: move = 'displace'
            elif r < self.p_displace + self.p_change: move = 'change'
            elif r < self.p_displace + self.p_change + self.p_remove: move = 'remove'
            else: move = 'insert'
            
            print(f"--- Step {step}/{self.iterations} | Move: {move.upper()} ---")
            new_struct, delta_N, valid_move = self.attempt_move(structure, move)
            
            accepted = False
            delta_e_val = 0.0
            new_energy = current_energy
            
            if not valid_move:
                print("Invalid/Impossible move. Skipped evaluation.\n")
            else:
                self.write_poscar(new_struct)
                self.evaluate_energy(step)
                new_energy_eval = self.read_energy()
                
                if new_energy_eval is not None:
                    delta_e_val = new_energy_eval - current_energy
                    if self.metropolis_acceptance(delta_e_val, delta_N):
                        accepted = True
                        new_energy = new_energy_eval

            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, move, valid_move, accepted, f"{delta_e_val:.4f}", f"{new_energy:.4f}"])
                f.flush()

            if accepted:
                print(f"Accepted! dE: {delta_e_val:.3f} eV\n")
                current_energy = new_energy
                structure = Structure.from_file(self.contcar_path)
                shutil.copy(self.contcar_path, os.path.join(self.traj_dir, f"step_{step}_CONTCAR"))
            else:
                if valid_move:
                    print("Rejected.\n")
                self.write_poscar(structure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GCMC Surface Sampler")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input', type=str, default='POSCAR')
    args = parser.parse_args()
    GCMCSampler(args.config).execute_gcmc_loop(args.input)