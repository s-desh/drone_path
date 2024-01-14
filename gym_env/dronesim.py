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
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os

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
                 detect_obstacle=False,
                 # If false, all obstacles known to drone, else drone detects obstacle in _detect_obstacles method.
                 show_progress=True
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

        self.detected_object_ids = []
        self.radius_cyl = 0.5
        self.height_cyl = 2.0
        self.obstacle_detect_threshold = 0.5
        self.cylinder_posns = None
        self.detect_obstacle = detect_obstacle
        self.show_progress = show_progress
        self.color_progress = [(0, int(x), int(x)) for x in np.linspace(100, 200, num_drones)]

        if initial_xyzs is None:
            initial_xyzs, self.cylinder_posns = self.create_env(num_drones, num_cylinders)
            initial_rpys = np.zeros_like(initial_xyzs)

        # For OCC map
        self.drone_size = (.06 * 1.15, .025)  # in meters. Represented as cylinder (radius * additional space & height)
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
                  (
                      int(self.drone_size_reduced[0] * self.resolution),
                      int(self.drone_size_reduced[0] * self.resolution)),
                  int(self.drone_size_reduced[0] * self.resolution), 255, -1)

        self.world_map_hidden = np.zeros((area_size * self.resolution, area_size * self.resolution),
                                         dtype=np.uint8)  # Track obstacles. 0 -> Free space; 1 -> obstacle
        self.occ_map_red = None  # For control
        self.occ_map = None  # For finding path
        self.occ_map_red_hidden = None  # For control
        self.occ_map_hidden = None  # All obstacles
        self.progress_map = None

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
        return

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
            return value / self.resolution - self.area_size / 2
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
        print("Environment Planning complete")
        return drone_nodes_xyz, cylinder_nodes_xyz

    def _detectObstacles(self):
        thresh = self.obstacle_detect_threshold
        if self.detect_obstacle:
            for drone in range(self.NUM_DRONES):
                pos = self.pos[drone, :2]
                min_x, max_x = np.clip(self.meter_to_world_map(pos[1] - thresh), 0, self.occ_map_hidden.shape[1]), \
                               np.clip(self.meter_to_world_map(pos[1] + thresh), 0, self.occ_map_hidden.shape[1])
                min_y, max_y = np.clip(self.meter_to_world_map(pos[0] - thresh), 0, self.occ_map_hidden.shape[0]), \
                               np.clip(self.meter_to_world_map(pos[0] + thresh), 0, self.occ_map_hidden.shape[0])

                self.occ_map[min_x: max_x, min_y: max_y] = self.occ_map_hidden[min_x: max_x, min_y: max_y]
                self.occ_map_red[min_x: max_x, min_y: max_y] = self.occ_map_red_hidden[min_x: max_x, min_y: max_y]

            cv.imshow("occ_map", self.occ_map)
            cv.imshow("occ_map_hidden", self.occ_map_hidden)
        else:
            pass
            # Do nothing
        if self.show_progress:
            for drone in range(self.NUM_DRONES):
                pos = self.pos[drone, :2]
                curr_drone_pos = self.meter_to_world_map(pos)
                cv.circle(self.progress_map, curr_drone_pos, radius=1, color=self.color_progress[drone], thickness=-1)
            cv.imshow("preview_map", cv.rotate(self.progress_map, cv.ROTATE_90_COUNTERCLOCKWISE))
            path = os.path.join(self.IMG_PATH, "preview_map_frame_" + str(self.FRAME_NUM) + ".png")
            cv.imwrite(path, cv.rotate(self.progress_map, cv.ROTATE_90_COUNTERCLOCKWISE))
        key = cv.waitKey(1)
        if key == ord('q'):
            cv.destroyAllWindows()
        return

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

        robot = ET.Element("robot", name="wall.urdf")

        # Create the link element
        link = ET.SubElement(robot, "link", name="wall")

        # Create the contact element
        contact = ET.SubElement(link, "contact")
        ET.SubElement(contact, "lateral_friction", value="1.0")
        ET.SubElement(contact, "rolling_friction", value="0.0")
        ET.SubElement(contact, "contact_cfm", value="0.0")
        ET.SubElement(contact, "contact_erp", value="1.0")

        # Create the inertial element
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", rpy="0 0 0", xyz="0.0 0.0 0.0")
        ET.SubElement(inertial, "mass", value="1")
        inertia = ET.SubElement(inertial, "inertia", ixx="0.0833", ixy="0.0", ixz="0.0", iyy="0", iyz="0.0", izz="0")

        # Create the visual element
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", rpy="0 0 0", xyz="0 0 0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", size=".5 32.5 1")
        material = ET.SubElement(visual, "material", name="beige")
        ET.SubElement(material, "color", rgba="1 0.77647058823 0.6 1")

        # Create the collision element
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz="0 0 0")
        ET.SubElement(collision, "geometry").append(ET.SubElement(geometry, "box", size=".5 32.5 1"))

        # Create the XML tree
        tree = ET.ElementTree(robot)
        
        # Save the URDF to a file
        urdf_file_path = "wall_urdf.xml"
        tree.write(urdf_file_path)
		
        p.loadURDF(urdf_file_path,
                   [16.25, -0.25, 0.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
                   
        p.loadURDF("assets/wall.urdf",
           [0.25, 16.25, 0.5],
           p.getQuaternionFromEuler([0, 0, 1.57079633]),
           physicsClientId=self.CLIENT
           )
           
        p.loadURDF("assets/wall.urdf",
           [-0.25, -16.25, 0.5],
           p.getQuaternionFromEuler([0, 0, 1.57079633]),
           physicsClientId=self.CLIENT
           )
        p.loadURDF("assets/wall.urdf",
           [-16.25, 0.25, 0.5],
           p.getQuaternionFromEuler([0, 0, 0]),
           physicsClientId=self.CLIENT
           )
                                                           
        
        
        if self.cylinder_posns is None:
            for _ in tqdm(range(self.num_cylinders), "Spawning cylinders"):
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
                cv.circle(self.world_map_hidden, (self.meter_to_world_map(x_cyl), self.meter_to_world_map(y_cyl)),
                              int(radius_cyl * self.resolution), 255, -1)

        else:
            for posn in self.cylinder_posns:
                self.cylinder_object_ids.append(p.loadURDF("assets/cylinder.urdf",
                                                           posn,
                                                           p.getQuaternionFromEuler([0, 0, 0]),
                                                           physicsClientId=self.CLIENT
                                                           ))
                cv.circle(self.world_map_hidden, (self.meter_to_world_map(posn[0]), self.meter_to_world_map(posn[1])),
                              int(self.radius_cyl * self.resolution), 255, -1)

        self.occ_map_hidden = create_occ_map(self.world_map_hidden, self.drone_obs_matrix)
        self.occ_map_red_hidden = create_occ_map(self.world_map_hidden, self.drone_obs_matrix_red)
        if not self.detect_obstacle:
            self.occ_map = self.occ_map_hidden
            self.occ_map_red = self.occ_map_red_hidden
        else:
            self.occ_map = np.zeros_like(self.occ_map_hidden)
            self.occ_map_red = np.zeros_like(self.occ_map_red_hidden)
        self.progress_map = np.zeros(list(self.world_map_hidden.shape) + [3])
        self.progress_map[:, :, 0] = self.world_map_hidden
        print("Obstacles Added")
        return
