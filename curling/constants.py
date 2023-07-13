import numpy as np

from dataclasses import dataclass, field
from typing import List, Union

from .utils import classproperty
from .enums import Accuracy

@dataclass
class SimulationConstants:
    time_intervals: Union[np.floating, np.ndarray] = (.3, .1, .05)
    num_points_on_circle: np.integer = field(default_factory=lambda: np.array(20))
    eps: np.floating = field(default_factory=lambda: np.array(1e-6))
    accuracy: Accuracy = Accuracy.LOW
    def __post_init__(self):
        # convert to numpy array
        self.time_intervals = np.array(self.time_intervals)

    @property
    def dtheta(self) -> np.floating:
        """angle between points on circle"""
        return 2 * np.pi / self.num_points_on_circle

    @property
    def dt(self) -> np.floating:
        """time interval for simulation"""
        if self.time_intervals.ndim == 0:
            # if time_intervals is a scalar, use that value
            return self.time_intervals
        # otherwise, use the value corresponding to the current accuracy
        return self.time_intervals[min(self.accuracy, len(self.time_intervals) - 1)]

    def reset(self):
        """resets accuracy to low"""
        self.accuracy = Accuracy.LOW

@dataclass
class PhysicalConstants:
    mu_0: float = 0.008 # coefficient of friction
    A: float = 10. # ratio of differences between rear and front coefficients of friction
    g: float = 9.81 # acceleration of freefall
    @classproperty
    def k(cls) -> np.floating:
        return cls.A / (cls.A + 1) # linear factor for scaling coefficient of friction

    @classmethod
    def calculate_friction(cls, speed: Union[float, List[float]], position: Union[float, List[float]] = 0) -> float:
        """calculates friction using inverse sqrt law

        Args:
            speed (Union[float, List[float]]): speed relative to ground
            position (Union[float, List[float]], optional): position in [-1, 1] relative to the front of the puck (1 is the front). Defaults to 0.

        Returns:
            float: coefficient of friction for point on stone
        """
        return np.maximum(cls.mu_0 / np.sqrt(speed) * (1 - cls.k * position), 0)
