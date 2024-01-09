import numpy as np
import pandas as pd
import tqdm

from RRT import RRTStar
import cv2 as cv
import time

radius_cyl = 0.5
resolution = 100


# 0.22956841138 -> basal area

def get_random_posn(area_size):
    out = np.random.random(2) * area_size - area_size / 2
    return out


def get_num_of_trees(area_size):
    n = .19 * (area_size * area_size) / (np.pi * radius_cyl * radius_cyl)
    return n


def meter_to_world_map(area_size, value):
    if isinstance(value, float):
        return int((value + area_size / 2) * resolution)
    elif isinstance(value, np.ndarray):
        value_shape = value.shape
        value_flatten = value.flatten()
        n_value_flatten = np.empty_like(value_flatten, dtype=int)
        for i in range(len(value_flatten)):
            n_value_flatten[i] = meter_to_world_map(area_size, value_flatten[i])
        return n_value_flatten.reshape(value_shape)
    else:
        raise ValueError("Unsupported data type for 'value' parameter.")


def create_occ_map(area_size, num_of_cylinders: int, random_goals=True):
    drone_nodes = np.empty((2, 2), dtype=float)
    drone_nodes[0] = np.array([-area_size / 2 * .9, -area_size / 2 * .9])
    if not random_goals:
        drone_nodes[1] = np.array([+area_size / 2 * .9, +area_size / 2 * .9])
    else:
        drone_nodes[1] = drone_nodes[:, 1]

    cylinder_nodes = np.empty((num_of_cylinders, 2), dtype=float)

    cylinder_nodes[:] = drone_nodes[0]
    world_map_hidden = np.zeros((area_size * resolution, area_size * resolution), dtype=np.uint8)

    for i in range(num_of_cylinders):
        while True:
            temp = get_random_posn(area_size)
            dist = min(np.min(np.linalg.norm(cylinder_nodes - temp, axis=1)),
                       np.min(np.linalg.norm(drone_nodes - temp, axis=1)))
            if dist > radius_cyl * 2 * 1.15:
                break
        cylinder_nodes[i] = temp
        cv.circle(world_map_hidden, (meter_to_world_map(area_size, temp[0]), meter_to_world_map(area_size, temp[1])),
                  int(radius_cyl * resolution), 255, -1)

    if random_goals:
        while True:
            goal = meter_to_world_map(area_size, get_random_posn(area_size))
            if world_map_hidden[goal[1], goal[0]] == 0:
                break
    else:
        goal = meter_to_world_map(area_size, drone_nodes[1])

    ratio = (np.sum(world_map_hidden) / 255) / (world_map_hidden.shape[0]*world_map_hidden.shape[0])
    start = meter_to_world_map(area_size, drone_nodes[0])
    return world_map_hidden, start, goal, cylinder_nodes, ratio


def run_RRT(area_size, num_cyll, runs=30):
    output = []
    for i in tqdm.tqdm(range(runs), desc=str(area_size)):
        occ_map, start, goal, cyl_node, ratio = create_occ_map(area_size, num_cyll, False)

        st = time.time()
        rrt = RRTStar(occ_map.copy(), start, goal, 20, 5000, False)
        rrt.find_path()
        et = time.time()
        cv.imshow('rrt', rrt.plot_graph())
        cv.waitKey(1)
        output.append([area_size, num_cyll, ratio, et - st])
    return output


if __name__ == '__main__':
    area_sizes = [3, 5, 10]
    output = []
    for ars in area_sizes:
        num_cyl = int(get_num_of_trees(ars))
        output += run_RRT(ars, num_cyl)

    df = pd.DataFrame(output, columns=['area size (m)', 'number of cyllinder', 'ratio', 'time'])
    df = df.reset_index()
    df.to_csv('RRT discsussion.csv')

