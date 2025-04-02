#!/bin/bash
# Migration script to update directory structure

# Copy base LBM files
mkdir -p shitty_bird/base/lbm
cp -r src/shitty_bird/core/lbm/* shitty_bird/base/lbm/
cp src/shitty_bird/core/simulation.py shitty_bird/base/
touch shitty_bird/base/__init__.py
touch shitty_bird/base/lbm/__init__.py

# Copy simulation files
mkdir -p shitty_bird/simulations
cp -r src/shitty_bird/simulations/* shitty_bird/simulations/
touch shitty_bird/simulations/__init__.py

# Copy utility files
mkdir -p shitty_bird/utils
cp -r src/shitty_bird/utils/* shitty_bird/utils/
touch shitty_bird/utils/__init__.py

# Copy main package __init__.py
cp src/shitty_bird/__init__.py shitty_bird/

# Create empty directories to ensure structure is complete
mkdir -p shitty_bird/tests

echo "Migration complete! You can now install using:"
echo "poetry install"
echo ""
echo "To verify the installation, run:"
echo "poetry run pytest"