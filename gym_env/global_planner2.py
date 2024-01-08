import numpy as np

min_th = 0
max_th = np.pi / 2


def cart_to_polar(ind):
    r = np.linalg.norm(ind)
    theta = np.atan()


def global_path(grid_size, num_drones):
    nodes = np.array([[x+1, y+1] for x in range(grid_size) for y in range(grid_size)])

    nodes_polar_r = np.linalg.norm(nodes, axis=1)
    nodes_polar_th = np.arctan2(nodes[:, 0].astype(float), nodes[:, 1].astype(float))

    th_region = np.linspace(min_th, max_th, num_drones + 1)

    visit = np.ones((grid_size, grid_size))*-1
    drone_nodes_dict = {}
    for i in range(num_drones):
        drone_the = [th_region[i], th_region[i+1]]
        drone_inds = np.argwhere(np.bitwise_and(nodes_polar_th >= drone_the[0], nodes_polar_th < drone_the[1]))
        drone_r = np.vstack([drone_inds.flatten(), nodes_polar_r[drone_inds].flatten()]).T
        drone_r_inds = np.argsort(drone_r[:, 1]).flatten()
        final_inds = drone_r[drone_r_inds, 0].astype(int)
        drone_nodes = nodes[final_inds]
        drone_nodes_dict[i+1] = drone_nodes.copy()
        j = 1
        for dn in drone_nodes:
            visit[dn[0]-1, dn[1]-1] = (i + j / 100)*1000
            j += 1
    return visit, drone_nodes_dict


if __name__ == '__main__':
    visit, dron_dict = global_path(20, 5)

    for key, value in dron_dict.items():
        print(key, ": ", len(value))

    print("Done")
