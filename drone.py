import numpy as np
from gym_env.RRT import RRTStar

class Drone:
    def __init__(self, id, env, global_path):
        self.id = id
        self.env = env
        self.global_path = global_path

    def get_curr_posn(self, meter=True):
        posn = self.env.pos[self.id, :2]
        if meter:
            posn = self.env.meter_to_world_map(posn)
        return posn

    def update(self, occ_map):
        self.rrt = RRTStar(occ_map, self.get_curr_posn(), goals[i], 20, 5000, True, id)




    def __init_subclass__(cls, **kwargs):
