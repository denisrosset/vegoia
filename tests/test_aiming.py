import matplotlib.pyplot as plt

from vegoia import *


def test_aiming() -> None:
    f = lambda x, y: x * x + y * y
    n = 2**20
    delta = 2**10
    grid = Grid.make(-1.0, 1.0, -1.0, 1.0, n, n)
    data = Data.empty(n, n)

    def callback(x: float, y: float, v: float) -> None:
        plt.plot(x, y, "bx")
        plt.show(block=False)

    implicit = Implicit(f, grid, data, 0.5, callback, None)
    print(Lip.from_bisection(None, None, delta, implicit))
