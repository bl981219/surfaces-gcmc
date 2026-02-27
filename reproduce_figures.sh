#!/bin/bash
# reproduce_figures.sh

echo "Reproducing Bulk Phase Diagram..."
python src/bulk_thermo.py --config config.yaml

echo "Identifying Competing Phases from Trajectory Data..."
# Evaluates the JSON trajectory to find conditions where secondary phases precipitate
python src/competing_phases_identifier.py --config config.yaml --data data/mu_trajectory.json

echo "Reproducing Surface Phase Diagram..."
# (Assuming they downloaded the example VASP data into an 'examples' folder)
python src/surface_thermo.py --config config.yaml --data_dir ./examples/vasp_data/

echo "Done! Check the 'output/' directory for the generated PNG and TXT files."