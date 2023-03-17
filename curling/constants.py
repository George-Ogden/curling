import numpy as np

from dataclasses import dataclass
from typing import List, Union

from .enums import Accuracy

@dataclass
class SimulationConstants:
    time_intervals: Union[np.floating, np.ndarray] = np.array((.3, .1, .05))
    num_points_on_circle: np.integer = np.array(20)
    eps: np.floating = np.array(1e-6)
    accuracy: Accuracy = Accuracy.LOW
    def __post_init__(self):
        self.time_intervals = np.array(self.time_intervals)

    @property
    def dtheta(self) -> np.floating:
        return 2 * np.pi / self.num_points_on_circle

    @property
    def dt(self) -> np.floating:
        if self.time_intervals.ndim == 0:
            return self.time_intervals
        return self.time_intervals[min(self.accuracy, len(self.time_intervals) - 1)]

    def reset(self):
        self.accuracy = Accuracy.LOW

@dataclass
class PhysicalConstants:
    mu_0: np.floating = np.array(0.008) # coefficient of friction = m_0 / sqrt(|v|)
    A: np.floating = np.array(10.) # ratio of differences between rear and front coefficients of friction
    k: np.floating = np.array((A - 1) / (A + 1)) # linear factor for scaling coefficient of friction
    g: np.floating = np.array(9.81) # acceleration of freefall

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
