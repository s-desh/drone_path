import numpy as np
from RRT import RRTStar, create_occ_map, transform_occ_img
import cv2 as cv
from control.DSLPIDControl import DSLPIDControl
from enums import DroneModel
from enums import Physics
from log_config import setup_logger

logger = setup_logger(__name__)

DEFAULT_DRONES = DroneModel("cf2x")
class Drone:
    def __init__(self, id, env, global_path, drone_model, stub=False):
        self.id = id
        self.env = env  
        self.global_path = global_path if not stub else np.array([[500.0,500.0], [500.0, 600.0], [500.0, 700.0]])
        # np.array([[947.0,900.0], [500.0, 500.0], [200.0, 200.0]])
        self.control = DSLPIDControl(drone_model=drone_model)
        self.iter = 0
        self.stub = stub
        self.local_start_posn = None
        self.local_goal_posn = None
        self.last_goal_posn = None
        self.local_origin = None
        # self.get_next_globalgoal_posn = None
        logger.info("Drone {} initialized".format(self.id))

    def get_next_globalgoal_posn(self, meter_to_world=True, local_coordinates=False):
        for i in range(self.iter, len(self.global_path)):
            goal_posn = self.global_path[i]
            if not self.stub:
                goal_posn = self.env.meter_to_world_map(np.array([self.global_path[i][0] - self.env.area_size/2, self.global_path[i][1] - self.env.area_size/2]))
            # check for obstacles
            if self.env.occ_map[int(goal_posn[1]), int(goal_posn[0])] == 0:
                self.iter = i
                break
        goal_posn = self.global_path[self.iter]
        if meter_to_world and not self.stub:
            # reorient the goal position to world map
            goal_posn = self.env.meter_to_world_map(np.array([goal_posn[0] - self.env.area_size/2, goal_posn[1] - self.env.area_size/2]))
        return goal_posn

    def get_curr_posn(self, meter_to_world=True, xyz=True):
        posn = self.env.pos[self.id, :]
        if not xyz:
            posn = posn[:2]
        if meter_to_world:
            posn = self.env.meter_to_world_map(posn)
        return posn
    
    def get_local_occmap(self, occ_map, start, gaol):
        # get new local occupancy map based on current position
        cx, cy = start[0], start[1]
        gx, gy = gaol[0], gaol[1]
        window_size = max(abs(gx - cx), abs(gy - cy))
        buffer = 20
        x_max = min(cx + window_size + buffer, self.env.occ_map.shape[0])
        y_max = min(cy + window_size + buffer, self.env.occ_map.shape[1])
        x_min = max(0, cx - window_size - buffer)
        y_min = max(0, cy - window_size - buffer)
        self.local_start_posn = np.array([cx - x_min, cy - y_min])
        self.local_goal_posn = np.array([gx - x_min, gy - y_min])
        self.local_origin = np.array([x_min, y_min])
        logger.info(f"Drone {self.id} : Local occupancy map window: {x_min, x_max, y_min, y_max}")
        local_occ_map = occ_map[int(y_min):int(y_max), int(x_min):int(x_max)]
        logger.info(f"Drone {self.id} : Local occupancy map shape: {local_occ_map.shape}")
        return local_occ_map

    def update(self, occ_map) -> None:
        # create new rrt for new global goal position, update occ_map
        logger.info(f"Drone {self.id} : Updating RRT for new global goal position")
        next_global_goal_posn = self.get_next_globalgoal_posn()
        logger.info(f"Drone {self.id} : Next global goal position for drone {next_global_goal_posn}")
        local_occ_map = self.get_local_occmap(occ_map, self.get_curr_posn(xyz=False), next_global_goal_posn)

        logger.info(f"start posn local {self.local_start_posn}")
        logger.info(f"goal posn local {self.local_goal_posn}")

        self.rrt = RRTStar(local_occ_map, self.local_start_posn, self.local_goal_posn, 20, 5000, True, self.id)
        logger.info(f"Drone {self.id} : Finding path ... ")
        _ = self.rrt.find_path()
        newocc_map = create_occ_map(self.env.world_map, self.env.drone_obs_matrix_red)
        self.rrt.update_occmap(self.get_local_occmap(newocc_map, self.get_curr_posn(xyz=False), next_global_goal_posn))
        plot_rrt = self.rrt.plot_graph()
        cv.imwrite(f"rrt_{self.iter}.png", transform_occ_img(plot_rrt))
        # threading error
        # cv.imshow("occupancy: " + str(self.id), transform_occ_img(plot_rrt))
        # cv.waitKey(0)
        logger.info(f"Drone {self.id} : RRT updated")
        self.control.reset()

    def step_action(self, obs, debug=False) -> np.ndarray:
        # curr posn in local map
        curr_pos = self.get_curr_posn(xyz=False) - self.local_origin
        goal_posn = self.rrt.get_next_posn(curr_pos[:2])
        if goal_posn is None:
            goal_posn = self.last_goal_posn
        else:
            self.last_goal_posn = goal_posn
        goal_posn_mtr = self.env.world_map_to_meter(goal_posn.astype(np.int64))
        target_drone_xyz = np.array([goal_posn_mtr[0], goal_posn_mtr[1], self.get_curr_posn(meter_to_world=False)[2]])
        target_drone_rpy = self.env.rpy[self.id, :]
        action, _, _ = self.control.computeControlFromState(control_timestep=self.env.CTRL_TIMESTEP,
                                                                 state=obs,
                                                                 target_pos=target_drone_xyz,
                                                                 # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                 target_rpy=target_drone_rpy
                                                                 )
        
        if debug:
            logger.info(f"drone {self.id} : current_drone_posn: {curr_pos}, next_micro_goal_posn: {goal_posn}, next_local_goal_posn: {self.local_goal_posn}")
        
        
         # if current and global goal position are same, increment the global goal position
        if np.allclose(self.get_curr_posn(xyz=False), self.get_next_globalgoal_posn(), atol=2):
            logger.info(f"Drone {self.id} : Current and global goal position are same")

            if self.iter + 1 == len(self.global_path):
                logger.info(f"------------Drone {self.id} : reached end of global path------------")
                exit()
            else:
                # update rrt with new global goal position
                self.iter += 1
                self.update(create_occ_map(self.env.world_map, self.env.drone_obs_matrix))

        return action
