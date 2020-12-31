import numpy as np
import matplotlib.pyplot as plt
import reading_utils

# INPUT VALUE BETWEEN 0 - 600 *unsure of bounds
# DEFINES TIME POINT
frame = 300

# INPUT VALUE BETWEEN 1 - 1000 *unsure of bounds
# DEFINES SIMULATION NUMBER
sim = 4

pos, col = reading_utils._get_data(sim,frame);

x, y = np.split(pos,[-1],axis=1)
color = col
plt.scatter(x, y, s=5, c=color, alpha=0.5)
plt.show()
