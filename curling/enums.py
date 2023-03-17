from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

class StoneColor(IntEnum):
    RED: int = -1
    YELLOW: int = 1
    def __invert__(self) -> StoneColor:
        if self == StoneColor.RED:
            return StoneColor.YELLOW
        elif self == StoneColor.YELLOW:
            return StoneColor.RED

class SimulationState(Enum):
    FINISHED: bool = True
    UNFINISHED: bool = False

class Colors(Enum):
    """colors Enum in BGR"""
    WHITE = (255, 255, 255)
    BACKGROUND = (255, 255, 128)
    BLUE = (255, 0, 0)
    RED = (0, 0, 255)
    GRAY = (128, 128, 128)
    YELLOW = (0, 255, 255)

@dataclass
class LinearTransform:
    m: float = 1 # gradient
    c: float = 0 # intercept
    def __call__(self, value: float) -> float:
        return value * self.m + self.c

class DisplayTime(Enum):
    """display times and linear equations for calculating time"""
    REAL_TIME: LinearTransform = LinearTransform(1, -1)
    TWICE_SPEED: LinearTransform = LinearTransform(.5, -1)
    NO_LAG: LinearTransform = LinearTransform(0, 1)
    FOREVER: LinearTransform = LinearTransform(0, 0)

class Accuracy(IntEnum):
    LOW: int = 0
    MID: int = 1
    HIGH: int = 2