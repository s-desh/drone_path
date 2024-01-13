# Multi agent motion planning simulation

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Usage](#usage)


## Overview 

This repository simulates multi agent path planning using drones in a forest for search and rescue operations using RRT* algorithm.

The simulation uses pybullet for physics
and few methods from gym-pybullet-drones
modified to suit our needs. Important parameters
used for experimenting are - number of drones,
number of trees and area size. For the most part
we’ve used 3 drones for exploring a forest area
of 900m2 with 200 trees (obstacles).


## Quick Start
1. Make sure `pip` is installed in your system, else install it with: 
    ```
    sudo apt-get install python3-pip
    ```
2. Make sure your `pip` installation is up to date:
    ```
    pip3 install --upgrade pip
    ```
3. Make sure you have `venv` installed, else install it with:
    ```
    sudo pip3 install virtualenv 
    ```
4. Clone the master branch of the repository using git clone.
5. Navigate to the path where `drone_path` is installed and use:
    ```
    cd drone_path
    ```
4. Run the shell script in gym_env. This will install all dependencies and run the simulation **once**.
    ```
    cd gym_env
    ./run.sh 10 5 2
    ```
5. `deactivate` to deactivate the virtual environment

## Usage

You can directly launch a simulation run after the virtual environment is ready, using`main.py`

1. Run - `python3 main.py --area 10 --num_trees 5 --num_drones 2`. With these parameters, the entire simulation will run within 3-4 mins while the experiment uses ones that take much longer.

2. Run `python3 main.py -h` for available command line arguments.

Note, `--record_video true` must always be paired with `--gui false` since they can't run together.

