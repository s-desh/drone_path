def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


import numpy as np
import cv2 as cv
import matplotlib.colors as mcolors

if __name__ == '__main__':
    orange = np.array(hex_to_rgb(mcolors.TABLEAU_COLORS['tab:orange']))
    green = np.array(hex_to_rgb(mcolors.TABLEAU_COLORS['tab:green']))
    red = np.array(hex_to_rgb(mcolors.TABLEAU_COLORS['tab:red']))
    data = cv.imread("Report/images/preview_map_frame_11193.png")
    for co, val in zip([orange, green, red], [100, 150, 200]):
        c = np.array([0, val, val])
        mask = cv.inRange(data, c, c)
        data[mask > 0] = co[::-1]

    cv.imwrite('Report/images/preview_map_frame_11193.png', data)
