import numpy as np
from gymnasium import spaces
import pybullet as p
import cv2 as cv
from PIL import Image
from enums import DroneModel, Physics, ImageType
import random
import cv2 as cv
from RRT import *
from controlenv import CtrlAviary
from enums import DroneModel, Physics


class DroneSim(CtrlAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 num_cylinders=10,  # Control number of cylinders to be made,
                 area_size=5,  # Area of the environment for our use case.
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
        #### PDM Additions ###################
        self.num_cylinders = num_cylinders
        self.area_size = area_size
        self.cylinder_object_ids = []  # Required to track the position of cylinders.
        self.resolution = 100  # 1 meter equals 100 pixels on map
        self.world_map = np.zeros((area_size * self.resolution, area_size * self.resolution),
                                  dtype=np.uint8)  # Track obstacles. 0 -> Free space; 1 -> obstacle
        self.detected_object_ids = []
        self.radius_cyl = 0.5
        self.height_cyl = 2.0
        self.obstacle_detect_threshold = 3
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
        self.drone_size = \
            (.06 * 1.15, .025)  # in meters. Represented as a cylinder (radius * additional space and height)
        self.drone_obs_matrix = np.zeros(
            (int(self.drone_size[0] * 2 * self.resolution), int(self.drone_size[0] * 2 * self.resolution)),
            dtype=np.uint8)
        cv.circle(self.drone_obs_matrix,
                  (int(self.drone_size[0] * self.resolution), int(self.drone_size[0] * self.resolution)),
                  int(self.drone_size[0] * self.resolution), 255, -1)

        self.occ_map = create_occ_map(self.world_map, self.drone_obs_matrix)
        # out = test_occ_map(self.occ_map, self.world_map)


    def meter_to_world_map(self, value: float):
        if isinstance(value, float):
            return int((value + self.area_size / 2) * self.resolution)
        elif isinstance(value, np.ndarray):
            value_shape = value.shape
            value_flatten = value.flatten()
            n_value_flatten = np.empty_like(value_flatten, dtype=int)
            for i in range(len(value_flatten)):
                n_value_flatten[i] = self.meter_to_world_map(value_flatten[i])
            return n_value_flatten.reshape(value_shape)
        else:
            raise ValueError("Unsupported data type for 'value' parameter.")


    def _detectObstacles(self):
        # obstacle detection based on current postion of drones, runs after every step
        thresh = self.obstacle_detect_threshold

        # print(self.cylinder_object_ids)

        for obsid in self.cylinder_object_ids:
            pos, orient = p.getBasePositionAndOrientation(obsid, physicsClientId=self.CLIENT)
            x, y, z = pos

            for drone in range(self.NUM_DRONES):
                # dist b/w drone and obs
                dist = np.sqrt(np.sum(np.square(self.pos[drone, :] - pos)))

                if (dist < thresh) and (obsid not in self.detected_object_ids):
                    self.detected_object_ids.append(obsid)
                    cv.circle(self.world_map, (self.meter_to_world_map(x), self.meter_to_world_map(y)),
                              int(self.radius_cyl * self.resolution), 255, -1)
                    # one drone can only be close to one cylinder below the thresh
                    continue

        cv.imshow("occupancy", self.world_map)

        key = cv.waitKey(100)
        if key == ord('q'):
            cv.destroyAllWindows()

    def check_collision_before_spawn(self, x, y, z, radius, height):
        # Check for collisions with existing cylinders
        collision = False
        for i in range(p.getNumBodies(physicsClientId=self.CLIENT)):
            body_info = p.getBodyInfo(i, physicsClientId=self.CLIENT)
            if body_info[0].decode('UTF-8') == "cylinder":
                # Get the position of the existing cylinder
                pos, _ = p.getBasePositionAndOrientation(i, physicsClientId=self.CLIENT)
                # Check if there is a collision with the existing cylinder
                distance = ((pos[0] - x) ** 2 + (pos[1] - y) ** 2 + (pos[2] - z) ** 2) ** 0.5
                if distance < (radius + 1.0):  # Adjust the collision distance based on your requirements
                    collision = True
                    break

        return collision

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        for _ in range(self.num_cylinders):
            while True:
                # Random position within the area
                x_cyl = random.uniform(-self.area_size / 2, self.area_size / 2)
                y_cyl = random.uniform(-self.area_size / 2, self.area_size / 2)
                z_cyl = 2  # Assuming the ground is at z=0
                radius_cyl = 0.5
                height_cyl = 2.0

                # Check for collisions before spawning
                if not self.check_collision_before_spawn(x_cyl, y_cyl, z_cyl, radius_cyl, height_cyl):
                    break

            # Spawn the cylinder at the random position
            self.cylinder_object_ids.append(p.loadURDF("assets/cylinder.urdf",
                                                       [x_cyl, y_cyl, z_cyl],
                                                       p.getQuaternionFromEuler([0, 0, 0]),
                                                       physicsClientId=self.CLIENT
                                                       ))
            cv.circle(self.world_map, (self.meter_to_world_map(x_cyl), self.meter_to_world_map(y_cyl)),
                      int(radius_cyl * self.resolution), 255, -1)
        return
