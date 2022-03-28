import numpy as num
from matplotlib import pyplot as plt
from pyrocko.modelling import eikonal

km = 1000.
nx, ny = 1500, 500                # grid size
delta = 90*km / float(nx)         # grid spacing
source_x, source_y = 0.0, 15*km   # source position

# Indexing arrays
x = num.arange(nx) * delta - 2*km
y = num.arange(ny) * delta
x2 = x[num.newaxis, :]
y2 = y[:, num.newaxis]

# Define layers with different speeds, roughly representing a crustal model.
speeds = num.ones((ny, nx))
nlayer = ny // 5
speeds[0*nlayer:1*nlayer, :] = 2500.
speeds[1*nlayer:2*nlayer, :] = 3500.
speeds[2*nlayer:3*nlayer, :] = 5000.
speeds[3*nlayer:4*nlayer, :] = 6000.
speeds[4*nlayer:, :] = 8000.

# Seeding points have non-negative times. Here we simply set one grid node to
# zero. The solution to the eikonal equation is computed at all nodes where
# times < 0.
times = num.zeros((ny, nx)) - 1.0
times[int(round(source_y/delta)), int(round((source_x-x[0])//delta))] = 0.0

# Solve eikonal equation.
eikonal.eikonal_solver_fmm_cartesian(speeds, times, delta)

# Plot
fig = plt.figure(figsize=(9.0, 4.0))

axes = fig.add_subplot(1, 1, 1, aspect=1.0)
axes.contourf(x/km, y/km, times)
axes.invert_yaxis()
axes.contourf(x/km, y/km, speeds, alpha=0.1, cmap='gray')
axes.plot(source_x/km, source_y/km, '*', ms=20, color='white')
axes.set_xlabel('Distance [km]')
axes.set_ylabel('Depth [km]')
# fig.savefig('eikonal_example1.png')
plt.show()
