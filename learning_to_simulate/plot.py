import numpy as np
import matplotlib.pyplot as plt
import reading_utils
from mpl_toolkits import mplot3d

# INPUT 2 OR 3 AND NAME OF DATATYPE
# DEFINES DIMENSIONS OF GRAPH AND GRAPH TYPE
pltType = 3
name = "Sand-3D"

# INPUT VALUE BETWEEN 0 - 600 *unsure of bounds
# DEFINES TIME POINT
frame = 100

# INPUT VALUE BETWEEN 1 - 1000 *unsure of bounds
# DEFINES SIMULATION NUMBER
sim = 2

if pltType == 2:
    pos, col = reading_utils._get_data(sim,frame, name);

    x, y = np.split(pos,[-1],axis=1)
    color = col
    plt.scatter(x, y, s=5, c=color, alpha=0.5)
    plt.xlim(.1, .9)
    plt.ylim(.1, .9)
    plt.show()
else:
    pos, col = reading_utils._get_data(sim,frame, name);
    print(pos)

    x, y, z = np.hsplit(pos,3)
    color = col

    ax = plt.axes(projection ="3d")
    ax.scatter3D(x, y, z, s=5, c=color, alpha=0.5)
    plt.xlim(.1, .9)
    plt.ylim(.1, .9)
    plt.show()



