# This file implements the RRT* algorithm for a quadcopter in 2D space.
# 2 points into the future along with current position are taken as input.
# Occupancy map is available via a class variables
import numpy as np
import itertools
import networkx as nx
from tqdm import tqdm
import cv2 as cv
from skimage.draw import line

inf = np.inf


def create_occ_map(world_map, drone_occupancy_map):
    pad_size = drone_occupancy_map.shape[0] // 2
    padded_map = cv.copyMakeBorder(world_map, pad_size, pad_size, pad_size, pad_size, cv.BORDER_CONSTANT,
                                   value=255)
    out_map = np.zeros_like(world_map)
    for i in range(out_map.shape[0]):
        for j in range(out_map.shape[1]):
            out_map[i, j] = np.sign(np.sum(np.multiply(padded_map[i:i + 2 * pad_size + 1, j:j + 2 * pad_size + 1],
                                                       drone_occupancy_map))) * 255
    return out_map


def test_occ_map(occ_map, world_map):
    out = np.zeros((list(occ_map.shape) + [3]))
    out[:, :, 0] = occ_map
    out[:, :, 1] = world_map
    return out


class Node:
    # For performing RRT in 2D space.
    id_iter = itertools.count()
    start = None

    def __init__(self, posn: np.ndarray, cost: float):
        self.posn = posn.astype(int)
        self.cost = cost
        self.id = next(self.id_iter)
        self.heuristic_cost = 0 if self.start is None else self - self.start
        self.parent_node = None
        self.child_nodes = []

    def __sub__(self, other):
        return np.sum((self.posn - other.posn) ** 2)

    def __str__(self):
        return f"X: {self.posn[0]} | y: {self.posn[1]} | Cost: {self.cost} | Id: {self.id}"

    def dist(self, other: np.ndarray):
        return np.sum((self.posn - other) ** 2)

    def set_prev_node(self, id: int):
        self.prev_node = id
        return

    def set_next_node(self, id: int):
        self.next_node = id
        return

    def set_parent(self, parent):
        self.parent_node = parent
        self.cost = parent.cost + (parent - self)
        for child in self.child_nodes:
            child.set_parent(self)

    def add_child(self, child_node):
        self.child_nodes.append(child_node)
        return

    def remove_child(self, child_node):
        self.child_nodes.remove(child_node)
        return


class RRTStar:
    def __init__(self,
                 occ_map: np.ndarray,  # Occupancy map (should be already expanded for collisions)
                 start: np.ndarray,  # Start config
                 goal: np.ndarray,  # Goal config
                 radius: float,  # Radius for finding close configurations
                 max_iter,  # Max iterations for RRT*
                 verbosity=True  # Show progress and other information
                 ):
        self.occ_map = occ_map

        self.graph = nx.Graph()
        self.start = Node(start, 0)
        Node.start = self.start
        self.add_node(self.start)
        self.goal = Node(goal, inf)  # Goal has not been added to the graph

        self.radius = radius
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.path_found = False
        self.find_path()

    @staticmethod
    def distance_func(n1, n2):
        return n1 - n2

    def add_node(self, node: Node):
        self.graph.add_node(node.id, item=node)
        return

    def add_edge(self, parent: Node, child: Node):
        self.graph.add_edge(parent.id, child.id, weight=parent - child)
        parent.add_child(child)
        child.set_parent(parent)
        return

    def replace_edge(self, parent: Node, child: Node, new_parent: Node):
        self.graph.remove_edge(parent.id, child.id)
        parent.remove_child(child)
        self.add_edge(new_parent, child)
        return

    def random_point(self):
        while True:
            ind = []
            shape = self.occ_map.shape
            for dim in shape:
                ind.append(np.random.randint(0, dim))
            if self.occ_map[ind[0], ind[1]] == 0:
                return np.array(ind)

    def line_collision(self, line: np.ndarray):
        # Check if the edge passes through an obstacle. Line is represented by a set of integer indices.
        # Returns True if their is collision
        assert line.dtype == int, "line is a list of indices and must be of type integers"
        coll = np.sum(self.occ_map[line[:, 0], line[:, 1]])
        return False if coll == 0 else True

    def nearest_node(self, new_node):
        # Get the nearest node to the new node
        nodes_list = list(nx.get_node_attributes(self.graph, 'item').values())
        arg = np.argmin(np.array(nodes_list) - new_node)
        return nodes_list[arg]

    def near_nodes(self, new_node):
        # Get nodes within radius R of the new node
        nodes_list = np.array(list(nx.get_node_attributes(self.graph, 'item').values()))
        arg = np.argwhere(nodes_list - new_node <= self.radius)
        return nodes_list[arg].flatten()

    def steer_func(self, nearest: Node, new: Node):
        prev_posn = nearest.posn
        line, _ = self.line_btw_nodes(nearest, new)
        for new_posn in line:
            if self.occ_map[new_posn[0], new_posn[1]] != 0:
                return prev_posn
            elif nearest.dist(new_posn) > self.radius:
                return prev_posn
            else:
                prev_posn = new_posn
        return prev_posn

    def line_btw_nodes(self, node1: Node, node2: Node):
        ind_line = np.array(line(node1.posn[0], node1.posn[1], node2.posn[0], node2.posn[1])).T
        collision = self.line_collision(ind_line)
        return ind_line, collision

    @staticmethod
    def update_cost(min_node: Node, new_node: Node):
        new_node.cost = min_node.cost + (new_node - min_node)
        return new_node

    def connect_goal(self):
        x_nearest = self.nearest_node(self.goal)
        dist = x_nearest - self.goal
        if dist <= self.radius:
            self.add_node(self.goal)
            self.add_edge(x_nearest, self.goal)

    def plot_graph(self, _graph=None):
        node_radius = 1  # pixel == cm
        line_thickness = 1  # thickness == cm
        out_img = np.zeros([self.occ_map.shape[0], self.occ_map.shape[1], 3])
        out_img[:, :, 0] = self.occ_map
        graph = self.graph if _graph is None else _graph
        # All nodes
        all_nodes_dict = nx.get_node_attributes(graph, 'item')
        nodes_list = list(all_nodes_dict.values())
        for node in nodes_list:
            cv.circle(out_img, node.posn, node_radius, (0, 255, 0), -1)

        edge_list = list(self.graph.edges)
        for edge in edge_list:
            par_edge = edge[0]
            chi_edge = edge[1]
            cv.arrowedLine(out_img, all_nodes_dict[par_edge].posn, all_nodes_dict[chi_edge].posn, (0, 255, 0),
                           thickness=line_thickness)

        # Path Found
        if self.path_found:
            node = self.goal
            while node is not self.start:
                par_node = node.parent_node
                cv.circle(out_img, node.posn, node_radius, color=(0, 0, 255))
                cv.arrowedLine(out_img, par_node.posn, node.posn, color=(0, 0, 255), thickness=line_thickness)

        return out_img

    def find_path(self):
        for i in tqdm(range(self.max_iter), disable=not self.verbosity):
            x_rand = Node(self.random_point(), -1)
            x_nearest = self.nearest_node(x_rand)
            x_new = Node(self.steer_func(x_nearest, x_rand), cost=-1)
            path_line, collision = self.line_btw_nodes(x_nearest, x_new)
            if not collision:
                x_near_arr = self.near_nodes(x_new)
                x_min = x_nearest
                x_new.cost = x_min.cost + (x_new - x_min)
                for x_near in x_near_arr:
                    path_line, collision = self.line_btw_nodes(x_near, x_new)
                    new_cost = x_near.cost + (x_new - x_near)
                    if (not collision) and (new_cost < x_new.cost):
                        x_min = x_near
                        x_new.cost = new_cost
                self.add_node(x_new)
                self.add_edge(x_min, x_new)
                for x_near in x_near_arr:
                    path_line, collision = self.line_btw_nodes(x_near, x_new)
                    new_cost = x_new.cost + (x_new - x_near)
                    if (not collision) and (new_cost < x_near.cost):
                        self.replace_edge(x_near.parent_node, x_near, x_new)

            if i // 100 == 0 and not self.path_found:
                self.connect_goal()


def test_RRT_start():
    print("Start Test")
    world_map = np.load("Test/world_map.npy")
    drone_obs_matrix = np.load("Test/drone_obs_matrix.npy")
    occ_map = np.load("Test/occ_map.npy")
    rrt = RRTStar(occ_map, np.array([20, 450]), np.array([450, 20]), 7, 2500, True)
    out_img = rrt.plot_graph()
    print("Test complete. Exiting....")
    return


if __name__ == '__main__':
    test_RRT_start()
