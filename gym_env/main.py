from datetime import datetime

import numpy as np
import argparse

from enums import Physics
from dronesim import DroneSim
from RRT import RRTStar, create_occ_map, transform_occ_img
from global_planner import bfs_multi_drones
from global_planner2 import global_path

from control.DSLPIDControl import DSLPIDControl
from enums import DroneModel
from drone import Drone
from log_config import setup_logger

logger = setup_logger(__name__)

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 2
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
DETECT_OBSTACLE = False
NUM_OF_CYLINDERS = 10
AREA_SIZE = 20
area_to_grid = 5


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
        num_trees=NUM_OF_CYLINDERS,
        area_size=AREA_SIZE,
        detect_obstacle=DETECT_OBSTACLE
):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = None
    INIT_RPYS = None

    #### Create the environment ################################
    logger.info("Creating environment...")
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
                   num_cylinders=num_trees,
                   area_size=area_size,
                   detect_obstacle=detect_obstacle
                   )

    #### Get global path #######################################
    GRID_SIZE = int(area_size / area_to_grid)
    path_plan, drone_paths = bfs_multi_drones(GRID_SIZE, num_drones)
    logger.info(f"Drone paths: {drone_paths}")

    #### Initialize drones #####################################
    drones = [Drone(
        id=i,
        env=env,
        global_path=drone_paths[i + 1], area=area_size, area_to_grid=area_to_grid, drone_model=drone, stub=False
    ) for i in range(num_drones)]

    for drone in drones:
        logger.info(f"update rrt for drone {drone.id}")
        drone.update()

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))  # rpms for every motor
    goals_reached = np.array([False] * num_drones)

    try:
        for i in range(0, int(duration_sec * env.CTRL_FREQ)):
            obs, reward, terminated, truncated, info = env.step(action)
            for drone in drones:
                action[drone.id, :], goals_reached[drone.id] = drone.step_action(obs[drone.id], debug=False)
            if np.all(goals_reached):
                logger.info("All goals reached!")
                exit()
    except Exception as e:
        logger.exception(e)
        raise

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


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    
    parser = argparse.ArgumentParser(description='Path planning for drones in a forest simulation')
  
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--num_trees',          default=NUM_OF_CYLINDERS,       type=int,      help='Number of trees in the environment (default: True)', metavar='')
    parser.add_argument('--area_size',               default=AREA_SIZE,        type=int,           help='Length of bounding forest area (default: 20)', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
