import numpy as np

class RRTs:
    def __init__(self):
        self.current_path = None  # Stores the computed optimum path
        self.occupancy_map = None  # Store obstacles in 3D numpy grid

        def check_path_collision():
            return # nodes that have collisions
