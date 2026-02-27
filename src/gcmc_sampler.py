# src/gcmc_sampler.py
# Author: Mengren Liu

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Outcar
import yaml

def read_structure_and_energy(outcar_path, contcar_path):
    """
    Replaces os.popen grepping. Safely reads the relaxed structure and 
    final energy from VASP/M3GNet outputs.
    """
    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
    except Exception:
        # Fallback if checking M3GNet mock OUTCAR
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
            energy = float(lines[-1].split()[-1])
            
    structure = Structure.from_file(contcar_path)
    return structure, energy

def write_poscar_with_selective_dynamics(structure, filename="POSCAR", frozen_z_max=0.07, frozen_z_min=0.9):
    """
    Replaces hardcoded poscar_write. Automatically applies T T T to active surface 
    layers and F F F to bulk levels based on fractional z-coordinates.
    """
    selective_dynamics = []
    for site in structure:
        # If the atom is in the middle "bulk" region, freeze it
        if frozen_z_max < site.frac_coords[2] < frozen_z_min:
            selective_dynamics.append([False, False, False]) # F F F
        else:
            selective_dynamics.append([True, True, True])    # T T T
            
    poscar = Poscar(structure, selective_dynamics=selective_dynamics)
    poscar.write_file(filename)