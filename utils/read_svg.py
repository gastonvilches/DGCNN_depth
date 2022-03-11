# Install:
#     pip3 install svgpathtools
#     pip3 install svgwrite     # if this one isn't installed, svg2paths raises an error

from svgpathtools import svg2paths
import matplotlib.pyplot as plt
import numpy as np

def path2xy(path):
    x = []
    y = []
    for line in path:
        start, end = line
        if len(x) == 0:
            x.append(start.real)
            y.append(start.imag)
        x.append(end.real)
        y.append(end.imag)
    return np.array(x), np.array(y)


file = 'C:/Users/Gastón/Desktop/Gaston/CONICET/Datasets/Depth/chairs/svg_visible/1d2745e280ea2d513c8d0fdfb1cc2535/azi_37_elev_21_0001.svg'

paths, attributes = svg2paths(file)

fig = plt.figure(figsize=(10, 10))
for path in paths:
    x, y = path2xy(path)
    plt.plot(x, 540-y, 'k')
plt.ylim(0, 540)
plt.xlim(0, 540)
