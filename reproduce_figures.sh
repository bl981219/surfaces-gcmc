#!/bin/bash
# reproduce_figures.sh

echo "Reproducing Bulk Phase Diagram..."
python src/bulk_thermo.py --config examples/example_config.yaml

echo "Identifying Competing Phases from Trajectory Data..."
# Evaluates the JSON trajectory to find conditions where secondary phases precipitate
python src/competing_phases_identifier.py --config examples/example_config.yaml

echo "Surface Sampling"
# Runs the GCMC sampler on the provided test POSCAR file. This will generate a trajectory JSON file in the output directory
python src/gcmc_sampler.py --config examples/example_config.yaml --input examples/test_POSCAR

echo "Reproducing Surface Phase Diagram..."
# (Assuming they downloaded the example VASP data into an 'examples' folder)
python src/surface_thermo.py --config examples/example_config.yaml --data_dir ./examples/converged_structures/

echo "Done! Check the 'output/' directory for the generated PNG and TXT files."