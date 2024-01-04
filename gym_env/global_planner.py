#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:05:37 2023

@author: manupriyasingh
"""
import numpy as np
from collections import deque


class GlobalPlanner:
    def __init__(self, drone_id, start):
        self.id = drone_id
        self.position = start
        self.visited_cells = set()
        self.path = []

    def move(self, grid, queue, visited):
        x, y = self.position

        # Moves in 4 directions
        potential_moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for move in potential_moves:
            if ((0 <= move[0] < grid.shape[0]) and (0 <= move[1] < grid.shape[1]) and (move not in visited)):
                # Move to the new cell and mark it as visited.
                self.position = move
                self.visited_cells.add(move)
                self.path.append(move)
                queue.append((self.id, move))
                return True

        return False


def bfs_multi_drones(grid_size, num_drones):
    grid = np.zeros((grid_size, grid_size))

    starting_points = [(3 + i, 0) for i in range(2, 2 + num_drones)]
    drones = [GlobalPlanner(i, start=starting_points[i]) for i in range(num_drones)]

    # Initialize a queue for BFS.
    bfs_queue = deque()

    # Start the BFS from the initial positions of the drones.
    for drone in drones:
        bfs_queue.append((drone.id, drone.position))
        drone.visited_cells.add(drone.position)
        drone.path.append(drone.position)

    while bfs_queue:
        current_drone_id, current_position = bfs_queue.popleft()

        current_drone = drones[current_drone_id]
        current_drone.move(grid, bfs_queue, set.union(*(drone.visited_cells for drone in drones)))

        # Added min path length so that each drone should move somewhat similar cells
        min_path_length = min(len(drone.path) for drone in drones)
        for i, drone in enumerate(drones):
            if len(drone.path) < min_path_length:
                bfs_queue.append((i, drone.position))

    # Return the paths covered by each drone.
    paths = {f"Drone {i + 1}": drone.path for i, drone in enumerate(drones)}
    return paths


if __name__ == "__main__":
    grid_size = 10
    num_drones = 5

    # Run the BFS for multiple drones.
    drone_paths = bfs_multi_drones(grid_size, num_drones)

    # Checking the covered cells by printing grid
    visit = np.zeros((grid_size, grid_size))
    for drone, path in drone_paths.items():
        drone_id = drone[-1]
        for p in path:
            visit[p[0], p[1]] = drone_id

    if np.any(visit == 0):
        print("There is at least one cell not visited.")

    else:
        print("All cells have been visited.")

    print("Grid:")
    print(visit)
    # checking the number of cells covered by each drone (checking min_length condition previosuly added)
    for key, value in drone_paths.items():
        print(key, len([item for item in value if item]))
