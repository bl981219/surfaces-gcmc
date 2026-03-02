#!/bin/bash
# reproduce_figures.sh

echo "Reproducing Bulk Phase Diagram..."
surfaces-bulk-thermo --config examples/example_config_1073K.yaml

echo "Identifying Competing Phases from Trajectory Data..."
# Evaluates the JSON trajectory to find conditions where secondary phases precipitate
surfaces-competing-phases --config examples/example_config_1073K.yaml --input examples/mu_trajectory.json

echo "Surface Sampling"
# Runs the GCMC sampler on the provided test POSCAR file. This will generate a trajectory JSON file in the output directory
surfaces-gcmc-sampler --config examples/example_config_1073K.yaml --input examples/test_POSCAR

echo "Reproducing Surface Phase Diagram..."
# (Assuming they downloaded the example VASP data into an 'examples' folder)
surfaces-surface-thermo --config examples/example_config_1073K.yaml --data_dir ./examples/converged_structures/
surfaces-surface-thermo --config examples/example_config_873K.yaml --data_dir ./examples/converged_structures/

echo "Done! Check the 'output/' directory for the generated PNG and TXT files."