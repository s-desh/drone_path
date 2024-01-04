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

HEIGHT_DIFF = .1  # Height difference between nodes


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
                 area_size=5.0,  # Area of the environment for our use case.
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
        self.obstacle_detect_threshold = 0.5
        self.cylinder_posns = None

        if initial_xyzs is None:
            initial_xyzs, self.cylinder_posns = self.create_env(num_drones, num_cylinders)
            initial_rpys = np.zeros_like(initial_xyzs)

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
        self.drone_size_reduced = (.06 * 1.05, .025)
        self.drone_obs_matrix = np.zeros(
            (int(self.drone_size[0] * 2 * self.resolution), int(self.drone_size[0] * 2 * self.resolution)),
            dtype=np.uint8)
        self.drone_obs_matrix_red = np.zeros(
            (int(self.drone_size[0] * 2 * self.resolution), int(self.drone_size[0] * 2 * self.resolution)),
            dtype=np.uint8)
        cv.circle(self.drone_obs_matrix,
                  (int(self.drone_size[0] * self.resolution), int(self.drone_size[0] * self.resolution)),
                  int(self.drone_size[0] * self.resolution), 255, -1)
        cv.circle(self.drone_obs_matrix_red,
                  (int(self.drone_size_reduced[0] * self.resolution), int(self.drone_size_reduced[0] * self.resolution)),
                  int(self.drone_size_reduced[0] * self.resolution), 255, -1)

    def meter_to_world_map(self, value):
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

    def world_map_to_meter(self, value):
        if isinstance(value, np.int64):
            return value / self.resolution - self.area_size/2
        elif isinstance(value, np.ndarray):
            value_shape = value.shape
            value_flatten = value.flatten()
            n_value_flatten = np.empty_like(value_flatten, dtype=float)
            for i in range(len(value_flatten)):
                n_value_flatten[i] = self.world_map_to_meter(value_flatten[i])
            return n_value_flatten.reshape(value_shape)
        else:
            raise ValueError("Unsupported data type for 'value' parameter.")

    def get_random_posn(self):
        out = np.random.random(2) * self.area_size - self.area_size / 2
        return out

    def create_env(self, num_of_drones: int, num_of_cylinders: int):
        drone_nodes = np.empty((num_of_drones, 2), dtype=float)
        cylinder_nodes = np.empty((num_of_cylinders, 2), dtype=float)

        for i in range(num_of_drones):
            drone_nodes[i] = np.array([-self.area_size / 2 * .9, -self.area_size / 2 * .9])

        cylinder_nodes[:] = drone_nodes[0]

        for i in range(num_of_cylinders):
            while True:
                temp = self.get_random_posn()
                dist = min(np.min(np.linalg.norm(drone_nodes - temp, axis=1)),
                           np.min(np.linalg.norm(cylinder_nodes - temp, axis=1)))
                if dist > self.radius_cyl * 2 * 1.15:
                    break
            cylinder_nodes[i] = temp
        drone_nodes_xyz = np.hstack(
            [drone_nodes, (np.arange(num_of_drones) + 1).reshape((num_of_drones, 1)) * HEIGHT_DIFF])
        cylinder_nodes_xyz = np.hstack([cylinder_nodes, np.ones((num_of_cylinders, 1)) * 1.1])
        return drone_nodes_xyz, cylinder_nodes_xyz

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

        # cv.imshow("occupancy", self.world_map.T)
        #
        # key = cv.waitKey(100)
        # if key == ord('q'):
        #     cv.destroyAllWindows()

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
        if self.cylinder_posns is None:
            for _ in range(self.num_cylinders):
                while True:
                    # Random position within the area
                    x_cyl = random.uniform(-self.area_size / 2, self.area_size / 2)
                    y_cyl = random.uniform(-self.area_size / 2, self.area_size / 2)
                    z_cyl = 2  # Assuming the ground is at z=0
                    radius_cyl = self.radius_cyl
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
                # todo: comment the line below
                cv.circle(self.world_map, (self.meter_to_world_map(x_cyl), self.meter_to_world_map(y_cyl)),
                          int(radius_cyl * self.resolution), 255, -1)
            return

        else:
            for posn in self.cylinder_posns:
                self.cylinder_object_ids.append(p.loadURDF("assets/cylinder.urdf",
                                                           posn,
                                                           p.getQuaternionFromEuler([0, 0, 0]),
                                                           physicsClientId=self.CLIENT
                                                           ))
                # todo: comment the line below
                cv.circle(self.world_map, (self.meter_to_world_map(posn[0]), self.meter_to_world_map(posn[1])),
                          int(self.radius_cyl * self.resolution), 255, -1)
