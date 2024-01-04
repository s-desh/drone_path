import numpy as np
from gym_env.RRT import RRTStar, create_occ_map, transform_occ_img
import cv2 as cv
from control.DSLPIDControl import DSLPIDControl
from enums import DroneModel
from enums import Physics

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
GRID_SIZE = int(AREA_SIZE / 5)

class Drone:
    def __init__(self, id, env, global_path):
        self.id = id
        self.env = env
        self.global_path = global_path
        self.control = DSLPIDControl(drone_model=DEFAULT_DRONES)
        self.iter = 0

    def get_goal_posn(self, meter=True):
        goal_posn = self.global_path[self.id]
        if meter:
            goal_posn = self.env.meter_to_world_map(goal_posn)
        return goal_posn

    def get_curr_posn(self, meter=True):
        posn = self.env.pos[self.id, :2]
        if meter:
            posn = self.env.meter_to_world_map(posn)
        return posn

    def update(self, occ_map) -> None:
        self.rrt = RRTStar(occ_map, self.get_curr_posn(), self.get_goal_posn(), 20, 5000, True, id)
        _ = self.rrt.find_path()
        newocc_map = create_occ_map(self.env.world_map, self.env.drone_obs_matrix_red)
        self.rrt.update_occmap(newocc_map)
        plot_rrt = self.rrt.plot_graph()
        cv.imshow("occupancy: " + str(self), transform_occ_img(plot_rrt))
        cv.waitKey(1)

    def step_action(self, obs) -> np.ndarray:
        curr_pos = self.get_curr_posn(meter=True)
        goal_posn = self.rrt.get_next_posn(curr_pos)
        goal_posn_mtr = self.env.world_map_to_meter(goal_posn)
        target_drone_xyz = np.array([goal_posn_mtr[0], goal_posn_mtr[1], curr_pos[2]])
        target_drone_rpy = self.env.rpy[self.id, :]
        action, _, _ = self.control.computeControlFromState(control_timestep=self.env.CTRL_TIMESTEP,
                                                                 state=obs[self.id],
                                                                 target_pos=target_drone_xyz,
                                                                 # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                 target_rpy=target_drone_rpy
                                                                 )

        return action
