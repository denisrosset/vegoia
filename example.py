import matplotlib.pyplot as plt

from vegoia import *

f = lambda x, y: x * x - y * y
n = 2**20
delta = 2**16
grid = Grid.make(-1.0, 1.0, -1.0, 1.0, n, n)
data = Data.empty(n, n)

fig = plt.figure()


def eval_callback(x: float, y: float, v: float) -> None:
    plt.plot(x, y, "bx")
    fig.show()
    fig.canvas.draw()
    plt.pause(0.001)


def line_callback(x1: float, y1: float, x2: float, y2: float) -> None:
    plt.plot([x1, x2], [y1, y2], "r-")


implicit = Implicit(f, grid, data, 0.5, eval_callback, line_callback)
# iso = Isoline.from_bisection(None, None, 0.1, implicit)
# lip, delta = Lip.from_bisection(implicit, None, None, delta)
# iso = Isoline.walk(lip, implicit)
isos = Isoline.from_grid(0.5, 0.1, implicit)
for iso in isos:
    x, y = iso.plot_coordinates()
    plt.plot(x, y, "k-")
fig.show()
plt.show()
