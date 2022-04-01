from __future__ import annotations

import logging
import math
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from re import X
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    overload,
)

import beartype.vale
import numpy as np
import numpy.typing as npt
import scipy.sparse
from numpy import linalg
from typing_extensions import Annotated

from .grid import Grid
from .isoline import Isoline
from .lip import Lip
from .square import Square, WalkResult
from .types import *

__all__ = [
    "Implicit",
    "Lip",
    "Data",
    "Grid",
    "Square",
    "Isoline",
]


class Fun(Protocol):
    def __call__(self, x: float, y: float) -> float:
        pass


@dataclass(frozen=True)
class Implicit:
    f: Fun
    grid: Grid
    data: Data
    altitude: float
    eval_callback: Optional[Callable[[float, float, float], None]]
    line_callback: Optional[Callable[[float, float, float, float], None]]

    def is_above(self, ix: int, iy: int) -> bool:
        return self.eval(ix, iy) >= self.altitude

    def is_below(self, ix: int, iy: int) -> bool:
        return self.eval(ix, iy) < self.altitude

    def eval(self, ix: int, iy: int) -> float:
        assert 0 <= ix <= self.grid.x_divs
        assert 0 <= iy <= self.grid.y_divs
        v = float(self.data.cache[ix, iy])
        if v == 0:
            x, y = self.grid.real_coordinates(ix, iy)
            f = self.f
            v = f(float(x), float(y))
            cb = self.eval_callback
            if cb:
                cb(x, y, v)
            if v == 0:
                v = sys.float_info.min
            self.data.cache[ix, iy] = v
        if v == sys.float_info.min:
            v = 0
        return v

    def find_integer_coordinates(
        self, x: float, y: float, above: bool, delta: int
    ) -> Tuple[int, int, int]:
        f = self.f
        ix0, iy0 = self.grid.integer_coordinates(x, y)
        c = self.eval(ix0, iy0) >= self.altitude
        if c == above:
            return ix0, iy0, delta
        while delta >= 1:
            for dx, dy in [(0, 1), (1, 0), (1, 1)]:
                ix = round(ix0 / delta + dx) * delta
                iy = round(iy0 / delta + dy) * delta
                c = self.eval(ix, iy) >= self.altitude
                if c == above:
                    return ix, iy, delta
            delta = delta // 2
        raise Exception("COuld not find starting point")


@dataclass(frozen=True)
class Data:
    #: Data array, type of elements is :data:`numpy.float64`
    cache: scipy.sparse.dok_array

    x_divs: int
    y_divs: int

    @staticmethod
    def empty(x_divs: int, y_divs: int) -> Data:
        cache = scipy.sparse.dok_array((x_divs + 1, y_divs + 1), dtype=np.float64)
        return Data(cache, x_divs, y_divs)
