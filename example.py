import matplotlib.pyplot as plt

from vegoia import *


def f(x: float, y: float) -> float:
    return x * x - y * y


n = 2**20
grid = Grid.make(-1.0, 1.0, -1.0, 1.0, n, n)
data = Data.empty(n, n)

fig = plt.figure()


# these two callbacks are so that we see progress, they are not needed


def eval_callback(x: float, y: float, v: float) -> None:
    plt.plot(x, y, "bx")
    fig.show()
    fig.canvas.draw()
    plt.pause(0.001)


def line_callback(x1: float, y1: float, x2: float, y2: float) -> None:
    plt.plot([x1, x2], [y1, y2], "r-")


# this means we plot the isoline f(x,y)=0.5
altitude = 0.5

# to not use callbacks, just do not provide them
implicit = Implicit.make(f, grid, data, altitude, eval_callback, line_callback)

# iso = Isoline.from_bisection(None, None, 0.1, implicit)
cell_size = 0.5  # size of the initial grid
plot_size = 0.1  # size of the line segments for the plot
isos = Isoline.from_grid(cell_size, plot_size, implicit)
for iso in isos:
    x, y = iso.plot_coordinates()
    plt.plot(x, y, "k-")
fig.show()
plt.show()
