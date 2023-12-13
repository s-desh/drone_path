

import numpy as np
from gymnasium import spaces
import pybullet as p
import cv2 as cv
from PIL import Image
from enums import DroneModel, Physics, ImageType
import random

from controlenv import CtrlAviary
from enums import DroneModel, Physics

class DroneSim(CtrlAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                         )

    

    def meter_to_world_map(self, value):
        return int((value + self.area_size/2)*self.resolution)

    def _detectObstacles(self):
        # obstacle detection based on current postion of drones, runs after every step
        thresh = self.obstacle_detect_threshold

        # print(self.cylinder_object_ids)

        for obsid in self.cylinder_object_ids:
            pos, orient = p.getBasePositionAndOrientation(obsid, physicsClientId=self.CLIENT)
            x, y, z = pos

            for drone in range(self.NUM_DRONES):
                # dist b/w drone and obs
                dist = np.sqrt(np.sum(np.square(self.pos[drone,:] - pos)))

                if (dist < thresh) and (obsid not in self.detected_object_ids):
                    self.detected_object_ids.append(obsid)
                    cv.circle(self.world_map, (self.meter_to_world_map(x), self.meter_to_world_map(y)), int(self.radius_cyl*self.resolution), 255, -1)
                    # one drone can only be close to one cylinder below the thresh
                    continue
                
        cv.imshow("occupancy",self.world_map)

        key = cv.waitKey(100)
        if key == ord('q'):
            cv.destroyAllWindows()
    