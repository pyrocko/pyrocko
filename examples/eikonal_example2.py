import numpy as num
from matplotlib import pyplot as plt
from pyrocko.modelling import eikonal

km = 1000.
nx, ny = 1000, 500          # grid size
delta = 20*km / float(nx)   # drid spacing

# Indexing arrays
x = num.arange(nx) * delta - 10.0
y = num.arange(ny) * delta
x2 = x[num.newaxis, :]
y2 = y[:, num.newaxis]

# Define layers and circles with different speeds, roughly representing a case
# with two layers and intrusions.
speeds = num.ones((ny, nx))
r1 = num.sqrt((x2-0*km)**2 + (y2-2*km)**2)
r2 = num.sqrt((x2-12*km)**2 + (y2-0*km)**2)
nlayer = ny // 5
speeds[r1 < 4*km] = 2.0
speeds[r2 < 4*km] = 0.7
speeds[:3*nlayer, :] *= 0.5

# Seeding points have non-negative times. Here we
times = num.zeros((ny, nx)) - 1.0
times[-1, :] = (x-num.min(x)) * 0.1

# Solve eikonal equation.
eikonal.eikonal_solver_fmm_cartesian(speeds, times, delta)

# Plot
fig = plt.figure(figsize=(9.0, 4.0))

axes = fig.add_subplot(1, 1, 1, aspect=1.0)
axes.contourf(x/km, y/km, times)
axes.invert_yaxis()
axes.contourf(x/km, y/km, speeds, alpha=0.1, cmap='gray')
axes.set_xlabel('Distance [km]')
axes.set_ylabel('Depth [km]')
# fig.savefig('eikonal_example2.png')
plt.show()
