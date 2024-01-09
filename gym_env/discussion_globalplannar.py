import numpy as np
import pandas as pd
import tqdm

from global_planner import bfs_multi_drones
from global_planner2 import global_path

import cv2 as cv
import time


def find_path_length(plannar, grid_size, num_drones):
    valid, drone_path = plannar(grid_size, num_drones)
    is_valid = not np.any(valid == -1)
    min_dist = grid_size*grid_size - 1
    nd = len(drone_path)
    dists = np.zeros((nd, 2))
    for i in range(nd):
        path = drone_path[i+1]
        cur = np.zeros(2)
        total_dist = 0
        for j in range(len(path)):
            dist = np.linalg.norm(path[j] - cur)
            cur = path[j]
            total_dist += dist
        dists[i] = np.array([i, total_dist])
    return is_valid, min_dist, np.sum(dists[:, 1])


if __name__ == '__main__':
    print(find_path_length(bfs_multi_drones, 6, 3))
    print(find_path_length(bfs_multi_drones, 10, 5))
    print(find_path_length(bfs_multi_drones, 10, 3))
    print(find_path_length(global_path, 6, 3))
    print(find_path_length(global_path, 10, 5))
    print(find_path_length(global_path, 10, 3))