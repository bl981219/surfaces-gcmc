import warnings

from m3gnet.models import Relaxer
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar


POSCAR_LSF = Poscar.from_file('POSCAR')
structure_POSCAR_LSF = POSCAR_LSF.structure

relaxer = Relaxer(relax_cell = False)  # This loads the default pre-trained model

relax_results = relaxer.relax(structure_POSCAR_LSF, verbose=True, steps = 10000)

final_structure = relax_results['final_structure']

final_structure.to('poscar','CONTCAR')

final_energy = float(relax_results['trajectory'].energies[-1])

with open("OUTCAR", "w") as output_file:
    output_file.write('  energy  without entropy=    ' + str(final_energy))

print(f"Relaxed lattice parameter is {final_structure.lattice.abc[0]:.3f} Å")
print(f"Final energy is {final_energy:.3f} eV")