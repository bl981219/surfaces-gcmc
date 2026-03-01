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
        self.change_species = self.config['gcmc_settings'].get('change_species', self.config['system']['elements'])
        self.min_dist = self.config['gcmc_settings'].get('min_distance', 0.5)
        
        self.z_gcmc_min, self.z_gcmc_max = self.config['gcmc_settings']['region_gcmc_z']
        self.z_mc_min, self.z_mc_max = self.config['gcmc_settings']['region_mc_z']
        
        self.vasp_cmd = self.config['gcmc_settings'].get('vasp_cmd', 'echo "Missing vasp cmd"')
        freq_val = self.config['gcmc_settings'].get('vasp_verification_freq', 'inf')
        self.vasp_freq = float('inf') if str(freq_val).lower() == 'inf' else int(freq_val)
        
        self.iterations = self.config['gcmc_settings']['iterations']
        
        self.p_displace = self.config['gcmc_settings'].get('displace_ratio', 0.05)
        self.p_exchange = self.config['gcmc_settings'].get('exchange_ratio', 0.10)
        self.p_change = self.config['gcmc_settings'].get('change_ratio', 0.05)
        self.p_remove = self.config['gcmc_settings'].get('remove_ratio', 0.40)
        self.p_insert = self.config['gcmc_settings'].get('insert_ratio', 0.40)
        
        p_tot = self.p_displace + self.p_exchange + self.p_change + self.p_remove + self.p_insert
        self.p_displace /= p_tot
        self.p_exchange /= p_tot
        self.p_change /= p_tot
        self.p_remove /= p_tot
        self.p_insert /= p_tot

        self.kb = 8.617333262145e-5

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

    def write_poscar(self, structure, path):
        selective_dynamics = [[self.z_mc_min <= site.coords[2] <= self.z_mc_max]*3 for site in structure]
        Poscar(structure, selective_dynamics=selective_dynamics).write_file(path)

    def prepare_vasp_files(self):
        """Searches for and copies required VASP input files to the working directory."""
        required_files = ['INCAR', 'POTCAR', 'KPOINTS']
        for vasp_file in required_files:
            if os.path.exists(vasp_file):
                shutil.copy(vasp_file, os.path.join(self.work_dir, vasp_file))
            else:
                raise FileNotFoundError(f"CRITICAL: Missing '{vasp_file}' in root directory for VASP verification!")

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
        action_details = f"Failed to perform {move_type}"
        
        if move_type == 'displace':
            active_idx = self.get_active_indices(new_struct, self.z_mc_min, self.z_mc_max)
            if active_idx:
                idx = random.choice(active_idx)
                original_coords = new_struct[idx].coords
                species = new_struct[idx].species_string
                for _ in range(10): 
                    frac_x, frac_y = random.random(), random.random()
                    cart_coords = new_struct.lattice.get_cartesian_coords([frac_x, frac_y, 0])
                    cart_coords[2] = random.uniform(self.z_mc_min, self.z_mc_max)
                    
                    new_struct.replace(idx, species, coords=cart_coords, coords_are_cartesian=True)
                    distances = new_struct.distance_matrix[idx]
                    distances[idx] = float('inf')
                    if np.min(distances) > self.min_dist:
                        valid_move = True
                        action_details = f"DISPLACE: Atom {idx} ({species}) moved to [{cart_coords[0]:.3f}, {cart_coords[1]:.3f}, {cart_coords[2]:.3f}]"
                        break
                if not valid_move:
                    new_struct.replace(idx, species, coords=original_coords, coords_are_cartesian=True)

        elif move_type == 'exchange':
            active_idx = self.get_active_indices(new_struct, self.z_mc_min, self.z_mc_max)
            if len(active_idx) >= 2:
                idx1, idx2 = random.sample(active_idx, 2)
                sp1, sp2 = new_struct[idx1].species_string, new_struct[idx2].species_string
                if sp1 != sp2:
                    new_struct.replace(idx1, sp2)
                    new_struct.replace(idx2, sp1)
                    valid_move = True
                    action_details = f"EXCHANGE: Swapped Atom {idx1} ({sp1}) with Atom {idx2} ({sp2})"

        elif move_type == 'change':
            active_idx = self.get_active_indices(new_struct, self.z_mc_min, self.z_mc_max, self.change_species)
            if active_idx:
                idx = random.choice(active_idx)
                old_species = new_struct[idx].species_string
                possible_new_species = [el for el in self.change_species if el != old_species]
                if possible_new_species:
                    new_species = random.choice(possible_new_species)
                    new_struct.replace(idx, new_species)
                    delta_N[old_species] = -1
                    delta_N[new_species] = 1
                    valid_move = True
                    action_details = f"CHANGE: Mutated Atom {idx} from {old_species} to {new_species}"

        elif move_type == 'remove':
            active_idx = self.get_active_indices(new_struct, self.z_gcmc_min, self.z_gcmc_max, self.gcmc_species)
            if active_idx:
                idx = random.choice(active_idx)
                el = new_struct[idx].species_string
                new_struct.remove_sites([idx])
                delta_N[el] = -1
                valid_move = True
                action_details = f"REMOVE: Deleted Atom {idx} ({el})"

        elif move_type == 'insert':
            insert_el = random.choice(self.gcmc_species)
            for _ in range(20):
                frac_x, frac_y = random.random(), random.random()
                cart_coords = new_struct.lattice.get_cartesian_coords([frac_x, frac_y, 0])
                cart_coords[2] = random.uniform(self.z_gcmc_min, self.z_gcmc_max)
                
                new_struct.insert(0, insert_el, cart_coords, coords_are_cartesian=True)
                distances = new_struct.distance_matrix[0]
                distances[0] = float('inf')
                if np.min(distances) > self.min_dist:
                    delta_N[insert_el] = 1
                    valid_move = True
                    action_details = f"INSERT: Added Atom ({insert_el}) at [{cart_coords[0]:.3f}, {cart_coords[1]:.3f}, {cart_coords[2]:.3f}]"
                    break
                else:
                    new_struct.remove_sites([0]) 
                    
        return new_struct, delta_N, valid_move, action_details

    def run_m3gnet(self, structure):
        """In-Memory Evaluation: passes structure directly to the ML model bypassing the disk."""
        if not self.relaxer:
            raise RuntimeError("M3GNet relaxer is not loaded.")
        relax_results = self.relaxer.relax(structure, verbose=False, steps=4000)
        final_structure = relax_results['final_structure']
        final_energy = float(relax_results['trajectory'].energies[-1])
        
        clean_struct = Structure(
            lattice=final_structure.lattice,
            species=final_structure.species,
            coords=final_structure.frac_coords,
            coords_are_cartesian=False
        )
        return final_energy, clean_struct

    def run_vasp(self, structure):
        """Executes VASP on disk and retrieves the DFT energy."""
        self.write_poscar(structure, self.poscar_path)
        self.prepare_vasp_files()
        exit_code = os.system(f"cd {self.work_dir} && {self.vasp_cmd}")
        if exit_code != 0:
            raise RuntimeError("VASP execution failed. Check vasp.out")
        return self.read_energy(), Structure.from_file(self.contcar_path)

    def execute_gcmc_loop(self, initial_poscar="POSCAR"):
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Move', 'Valid', 'M3G_Acc', 'VASP_Status', 'dE_M3G', 'Current_E'])

        structure = Structure.from_file(initial_poscar)
        
        # 1. Establish the true Ei,DFT Baseline
        if self.vasp_freq != float('inf'):
            print("\n--- Establishing Initial E_i,DFT Baseline ---")
            e_i_dft, structure = self.run_vasp(structure)
        else:
            print("\n--- Establishing Initial M3GNet Baseline ---")
            e_i_dft, structure = self.run_m3gnet(structure)
            
        print(f"Initial Baseline Energy: {e_i_dft:.4f} eV\n")
        
        # 2. Checkpoint variables for the VASP Rollback Mechanism
        checkpoint_struct = copy.deepcopy(structure)
        e_i_dft_checkpoint = e_i_dft
        cumulative_delta_N = {}
        
        # 3. Inner loop M3GNet energy tracking
        current_m3g_energy, _ = self.run_m3gnet(structure)

        for step in range(1, self.iterations + 1):
            r = random.random()
            if r < self.p_displace: move = 'displace'
            elif r < self.p_displace + self.p_exchange: move = 'exchange'
            elif r < self.p_displace + self.p_exchange + self.p_change: move = 'change'
            elif r < self.p_displace + self.p_exchange + self.p_change + self.p_remove: move = 'remove'
            else: move = 'insert'
            
            print(f"--- Step {step}/{self.iterations} | Move: {move.upper()} ---")
            new_struct, delta_N, valid_move, action_details = self.attempt_move(structure, move)
            
            m3g_accepted = False
            vasp_status = "N/A"
            delta_e_val = 0.0
            
            # --- TIER 1: M3GNet Speculative Sampling ---
            if valid_move:
                print(f"Proposed: {action_details}")
                new_m3g_energy, relaxed_struct = self.run_m3gnet(new_struct)
                delta_e_val = new_m3g_energy - current_m3g_energy
                
                if self.metropolis_acceptance(delta_e_val, delta_N):
                    m3g_accepted = True
                    structure = relaxed_struct
                    current_m3g_energy = new_m3g_energy
                    # Accumulate net atomic changes for the upcoming VASP check
                    for el, count in delta_N.items():
                        cumulative_delta_N[el] = cumulative_delta_N.get(el, 0) + count
                    print(f"M3GNet Speculative Accept! dE: {delta_e_val:.3f} eV")
                else:
                    print("M3GNet Rejected.")
            else:
                print(f"Invalid move. ({action_details})")

            # --- TIER 2: VASP Delayed Acceptance & Rollback ---
            if self.vasp_freq != float('inf') and step % self.vasp_freq == 0:
                print(f"\n[!] Triggering VASP Verification (Ef,DFT) at Step {step}...")
                e_f_dft, vasp_relaxed_struct = self.run_vasp(structure)
                
                delta_e_dft = e_f_dft - e_i_dft_checkpoint
                print(f"Ei,DFT: {e_i_dft_checkpoint:.4f} eV | Ef,DFT: {e_f_dft:.4f} eV | Delta E: {delta_e_dft:.4f} eV")
                print(f"Net Atom Change over {self.vasp_freq} steps: {cumulative_delta_N}")
                
                if self.metropolis_acceptance(delta_e_dft, cumulative_delta_N):
                    vasp_status = "ACCEPTED"
                    print(">>> VASP VERIFICATION ACCEPTED! <<<")
                    # Update Checkpoints
                    checkpoint_struct = copy.deepcopy(vasp_relaxed_struct)
                    e_i_dft_checkpoint = e_f_dft
                    
                    # Sync inner M3GNet loop back to the validated VASP structure
                    structure = copy.deepcopy(vasp_relaxed_struct)
                    current_m3g_energy, _ = self.run_m3gnet(structure)
                    cumulative_delta_N = {} 
                    
                    # Archive validated structure
                    step_dir = os.path.join(self.traj_dir, f"vasp_validated_step_{step}")
                    os.makedirs(step_dir, exist_ok=True)
                    self.write_poscar(structure, os.path.join(step_dir, "CONTCAR"))
                    with open(os.path.join(step_dir, "action.txt"), "w") as f:
                        f.write(f"VASP Validated Step {step}\nEf,DFT: {e_f_dft:.4f} eV")
                else:
                    vasp_status = "REJECTED (ROLLBACK)"
                    print(">>> VASP VERIFICATION REJECTED! Rolling back to last DFT checkpoint. <<<")
                    # Rollback the geometry and reset accumulators
                    structure = copy.deepcopy(checkpoint_struct)
                    current_m3g_energy, _ = self.run_m3gnet(structure)
                    cumulative_delta_N = {} 
            print("")

            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, move, valid_move, m3g_accepted, vasp_status, f"{delta_e_val:.4f}", f"{current_m3g_energy:.4f}"])
                f.flush()

def main():
    parser = argparse.ArgumentParser(description="Run Hybrid M3GNet/DFT GCMC Sampler")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input', type=str, default='POSCAR')
    args = parser.parse_args()
    GCMCSampler(args.config).execute_gcmc_loop(args.input)

if __name__ == "__main__":
    main()