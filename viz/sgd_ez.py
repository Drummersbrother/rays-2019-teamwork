import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d

np.random.seed(1)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
N = 15
step = 0.1
X, Y = np.meshgrid(np.arange(-N/2, N/2, step), np.arange(-N/2, N/2, step))
heights = (2+np.sin(X)) * (0.2 * (Y**2)) + X**2
ax.set_zlim3d(0, 100)
ax.plot_surface(X, Y, heights, cmap=plt.get_cmap('jet'), alpha=0.8)
for a in ax.w_xaxis.get_ticklines()+ax.w_xaxis.get_ticklabels():
    a.set_visible(False)
for a in ax.w_yaxis.get_ticklines()+ax.w_yaxis.get_ticklabels():
    a.set_visible(False)
for a in ax.w_zaxis.get_ticklines()+ax.w_zaxis.get_ticklabels():
    a.set_visible(False)

min_pos = [x*step for x in np.unravel_index(heights.argmin(), heights.shape)]
min_pos = [x - N/2 for x in min_pos]
print(min_pos)
ax.plot([min_pos[0]], [min_pos[1]], [heights.min()], c='r', marker='.', zorder=10)

plt.show()