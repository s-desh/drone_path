import numpy as np
from RRT import RRTStar, create_occ_map, transform_occ_img
import cv2 as cv
from control.DSLPIDControl import DSLPIDControl
from enums import DroneModel
from enums import Physics

DEFAULT_DRONES = DroneModel("cf2x")
class Drone:
    def __init__(self, id, env, global_path, drone_model, stub=False):
        self.id = id
        self.env = env  
        self.global_path = global_path if not stub else np.array([[947.0,900.0], [500.0, 500.0], [200.0, 200.0]])
        self.control = DSLPIDControl(drone_model=drone_model)
        # self.current_posn = self.env.pos[self.id, :2]
        self.iter = 0
        self.stub = stub
        print("Drone {} initialized".format(self.id))

    def get_next_globalgoal_posn(self, meter_to_world=True):
        # TODO: check if obstacle is present in the next global goal position
        goal_posn = self.global_path[self.iter]
        if meter_to_world and not self.stub:
            goal_posn = self.env.meter_to_world_map(np.array([goal_posn[0], goal_posn[1]]))              
        # print(f"Next global goal position {goal_posn}")
        return goal_posn

    def get_curr_posn(self, meter_to_world=True, xyz=True):
        posn = self.env.pos[self.id, :]
        if not xyz:
            posn = posn[:2]
        if meter_to_world:
            posn = self.env.meter_to_world_map(posn)
        return posn

    def update(self, occ_map) -> None:
        print("Updating RRT for new global goal position")
        self.rrt = RRTStar(occ_map, self.get_curr_posn(xyz=False), self.get_next_globalgoal_posn(), 20, 5000, True, id)
        _ = self.rrt.find_path()
        newocc_map = create_occ_map(self.env.world_map, self.env.drone_obs_matrix_red)
        self.rrt.update_occmap(newocc_map)
        plot_rrt = self.rrt.plot_graph()
        cv.imshow("occupancy: " + str(self), transform_occ_img(plot_rrt))
        cv.waitKey(1)

    def step_action(self, obs, debug=False) -> np.ndarray:
        curr_pos = self.get_curr_posn()
        goal_posn = self.rrt.get_next_posn(curr_pos[:2])
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
            print("current_drone_posn: ", curr_pos, "next_local_goal_posn: ", goal_posn, "next_global_goal: ", self.get_next_globalgoal_posn())
         # if current and global goal position are same, increment the global goal position
        if np.allclose(self.get_curr_posn(xyz=False), self.get_next_globalgoal_posn()):
            print("Current and global goal position are same")
            self.iter += 1
            if self.iter >= len(self.global_path):
                print("Reached the end of global path")
                exit()
            # update rrt with new global goal position
            self.update(create_occ_map(self.env.world_map, self.env.drone_obs_matrix))

        return action
