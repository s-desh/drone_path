"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories
in the X-Y plane, around point (0, -.3).

"""
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
from RRT import RRTStar, create_occ_map

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

# from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.utils.utils import sync, str2bool

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
AREA_SIZE = 5

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
        num_cyllinders = NUM_OF_CYLLINDERS,
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

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    # #### Initialize the controllers ############################
    # if drone in [DroneModel.CF2X, DroneModel.CF2P]:
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    # #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()
    goals = []
    for i in range(num_drones):
        while True:
            test_posn = env.meter_to_world_map(np.array([area_size/2 - np.random.random(),
                                                         area_size/2 - np.random.random()]))
            if env.occ_map[test_posn[1], test_posn[0]] == 0:
                goals.append(test_posn)
                break
    goals = np.array(goals)
    occ_map = create_occ_map(env.world_map, env.drone_obs_matrix)
    rrt_array = [RRTStar(occ_map, env.meter_to_world_map(env.pos[i, :2]), goals[i], 20, 5000, True, i) for i in range(num_drones)]
    for rrt in rrt_array:
        path = rrt.find_path()
        cv.imshow("occupancy: " + str(rrt), cv.rotate(rrt.plot_graph(), cv.ROTATE_90_CLOCKWISE))

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # print(i)
        obs, reward, terminated, truncated, info = env.step(action)
        for j in range(num_drones):
            curr_drone_pos_mtr = env.pos[j, :]
            curr_drone_pos = env.meter_to_world_map(curr_drone_pos_mtr[:2])
            goal_posn = rrt_array[j].get_next_posn(curr_drone_pos)
            goal_posn_mtr = env.world_map_to_meter(goal_posn)
            target_drone_xyz = np.array([goal_posn_mtr[0], goal_posn_mtr[1], curr_drone_pos_mtr[2]])
            target_drone_rpy = env.rpy[j, :]
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                 state=obs[j],
                                                                 target_pos=target_drone_xyz,
                                                                 # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                 target_rpy=target_drone_rpy
                                                                 )
            print("J: ", j, "current_drone_posn: ", curr_drone_pos, "goal_posn: ", goal_posn, "target_drone_posn: ",
                  target_drone_xyz)
        print(i)

    #### Close the environment #################################
    env.close()


def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)',
                        metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)',
                        metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
