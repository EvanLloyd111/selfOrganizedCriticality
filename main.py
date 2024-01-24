import numpy as np
import matplotlib.pyplot as plt


def lorenz(xyz, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives atxyz.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


dt = 0.01
num_steps = 10000
rs = [5, 15, 28]  #R values

#creates arrays for all Rs
for r in rs:
    xyzs = np.empty((num_steps + 1, 3))
    xyzs[0] = (0., 1., 1.05)  #intial values
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i], r=r) * dt

    # Plot x
    fig, ax = plt.subplots()
    ax.plot(xyzs[:, 0], lw=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("X Coordinate")
    ax.set_title(f"Lorenz Attractor: X Component (r = {r})")

    # Plot y
    fig, ax = plt.subplots()
    ax.plot(xyzs[:, 1], lw=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Lorenz Attractor: Y Component (r = {r})")

    # Plot z
    fig, ax = plt.subplots()
    ax.plot(xyzs[:, 2], lw=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Z Coordinate")
    ax.set_title(f"Lorenz Attractor: Z Component (r = {r})")

    # Plot 3D attractor
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor (r = {r})")

    ax.text2D(0.05, 0.95, f"r = {r}", transform=ax.transAxes)
    ax.text2D(0.05, 0.90, f"Final values: X = {xyzs[-1, 0]:.2f}, Y = {xyzs[-1, 1]:.2f}, Z = {xyzs[-1, 2]:.2f}", transform=ax.transAxes)
    ax.text2D(0.05, 0.85, f"Initial values: X = {xyzs[0, 0]:.2f}, Y = {xyzs[0, 1]:.2f}, Z = {xyzs[0, 2]:.2f}", transform=ax.transAxes)

#save figure
plt.savefig(f"lorenz_attractor_r={r}.png")

#show plot
plt.show()