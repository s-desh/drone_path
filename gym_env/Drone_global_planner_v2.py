#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:24:57 2023

@author: manupriyasingh
"""

class Drone:
    def __init__(self, drone_id, start=0):
        self.id = drone_id
        self.position = (0, start)  # Starting positions are different for each drone in different rows
        self.visited_cells = set()

    def move(self, grid_size):
        # Move the drone to the next unvisited cell in a systematic way.
        x, y = self.position
        new_position = (x + 1, y) if x + 1 < grid_size else (0, y + 1)
        self.position = new_position
        self.visited_cells.add(self.position)

def main(grid_size, num_drones):
    # Initialize grid and drones.
    grid = [[0] * grid_size for _ in range(grid_size)]
    # Start each drone at the beginning of each row.
    drones = [Drone(i, start=i) for i in range(num_drones)]

    # Number of cells in the grid.
    total_cells = grid_size * grid_size

    iteration = 0
    while sum(len(drone.visited_cells) for drone in drones) < total_cells:
        iteration += 1
        for drone in drones:
            drone.move(grid_size)

        # Print the current state if you want to observe the progress.
        print(f"Iteration {iteration}: {[{'id': drone.id, 'position': drone.position, 'visited_cells': len(drone.visited_cells)} for drone in drones]}")

    print("Simulation complete.")
    # Print the number of cells visited by each drone.
    for i, drone in enumerate(drones):
        print(f"Drone {i + 1} visited {len(drone.visited_cells)} cells.")

if __name__ == "__main__":
    # Set parameters for the simulation.
    grid_size = 50
    num_drones = 5

    # Run the simulation.
    main(grid_size, num_drones)