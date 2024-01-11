#!/bin/bash

echo "Creating virtual environment and installing dependencies..."
python -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt

echo "Ready, Set, Running simulation..."
python3 main.py --area "$1" --num_trees "$2" --num_drones "$3"

echo "Deactivating virtual environment..."
deactivate