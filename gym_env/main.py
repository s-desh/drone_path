import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2 as cv

from enums import Physics
from dronesim import DroneSim
from RRT import RRTStar, create_occ_map, transform_occ_img
from global_planner import bfs_multi_drones

from control.DSLPIDControl import DSLPIDControl
from enums import DroneModel
from drone import Drone

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 2000
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
NUM_OF_CYLLINDERS = 10
AREA_SIZE = 10
GRID_SIZE = int(AREA_SIZE / 1)

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        num_cyllinders=NUM_OF_CYLLINDERS,
        area_size=AREA_SIZE
):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = None
    INIT_RPYS = None

    #### Create the environment ################################
    env = DroneSim(drone_model=drone,
                   num_drones=num_drones,
                   initial_xyzs=INIT_XYZS,
                   initial_rpys=INIT_RPYS,
                   physics=physics,
                   neighbourhood_radius=10,
                   pyb_freq=simulation_freq_hz,
                   ctrl_freq=control_freq_hz,
                   gui=gui,
                   record=record_video,
                   obstacles=obstacles,
                   user_debug_gui=user_debug_gui,
                   num_cylinders=num_cyllinders,
                   area_size=area_size
                   )
    
    #### Get global path #######################################
    drone_paths = bfs_multi_drones(GRID_SIZE, num_drones)
    print(f"Drone paths: {drone_paths}")

    #### Initialize drones #####################################
    drones = [Drone(
        id=i,
        env=env,
        global_path=drone_paths[i+1], drone_model=drone, stub=True
    ) for i in range(num_drones)]

    occ_map = create_occ_map(env.world_map, env.drone_obs_matrix)
    for drone in drones:
        drone.update(occ_map)

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4)) # rpms for every motor    

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        for drone in drones:
            action[drone.id,:] = drone.step_action(obs[drone.id], debug=True)
    
    env.close()

if __name__ == '__main__':
    run()