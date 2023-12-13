#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:42:55 2023

@author: manupriyasingh
"""

import numpy as np
from collections import deque

class Drone:
    def __init__(self, drone_id, start=(50,0)):
        self.id = drone_id
        self.position = start
        self.visited_cells = set()
        self.path = []

    def move(self, grid, queue, visited):
        x, y = self.position

        # Define potential moves (up, down, left, right).
        potential_moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]

        for move in potential_moves:
            if (
                0 <= move[0] < grid.shape[0]
                and 0 <= move[1] < grid.shape[1]
                and move not in visited
            ):
                # Move to the new cell and mark it as visited.
                self.position = move
                self.visited_cells.add(move)
                self.path.append(move)
                queue.append((self.id, move))
                return True

        return False

def bfs_multi_drones(grid_size, num_drones):
    grid = np.zeros((grid_size, grid_size))
    drones = [Drone(i) for i in range(num_drones)]

    # Initialize a queue for BFS.
    bfs_queue = deque()

    # Start the BFS from the initial positions of the drones.
    for drone in drones:
        bfs_queue.append((drone.id, drone.position))
        drone.visited_cells.add(drone.position)
        drone.path.append(drone.position)

    while bfs_queue:
        current_drone_id, current_position = bfs_queue.popleft()

        # Move the current drone to an unvisited adjacent cell.
        current_drone = drones[current_drone_id]
        current_drone.move(grid, bfs_queue, set.union(*(drone.visited_cells for drone in drones)))

        # Continue the simulation until all drones have moved at least once.
        for i, drone in enumerate(drones):
            if len(drone.path) < len(current_drone.path):
                bfs_queue.append((i, drone.position))

    # Return the paths covered by each drone.
    paths = {f"Drone {i + 1}": drone.path for i, drone in enumerate(drones)}
    return paths

if __name__ == "__main__":
    # Set parameters for the simulation.
    grid_size = 100
    num_drones = 5

    # Run the BFS for multiple drones.
    drone_paths = bfs_multi_drones(grid_size, num_drones)
    
    ## Print the paths covered by each drone.
    for drone, path in drone_paths.items():
        print(f"{drone} path: {path}")
