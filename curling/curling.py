from __future__ import annotations

import numpy as np
import cv2

from typing import ClassVar, List, Optional, Tuple
from dataclasses import dataclass

from .constants import Accuracy, PhysicalConstants, SimulationConstants
from .enums import Colors, DisplayTime, StoneColor, SimulationState
from .stone import Stone

class Curling:
    physical_constants: PhysicalConstants = PhysicalConstants()
    pitch_length: np.floating = np.array(45.720)
    pitch_width: np.floating = np.array(4.750)
    hog_line_position: np.floating = np.array(11.888) # distance from back board to hog line
    tee_line_position: np.floating = np.array(5.487) # distance from back board to tee line
    back_line_position: np.floating = np.array(3.658) # distance from back board to back line
    button_position = np.array((0., -tee_line_position))
    starting_button_distance: np.floating = pitch_length - tee_line_position - hog_line_position # distance to button from where stone is released
    target_radii: np.ndarray = np.array((0.152, 0.610, 1.219, 1.829)) # radii of rings in the circle
    house_radius: np.floating = np.array((1.996)) # distance from centre of stone to button
    vertical_lines: np.ndarray = np.array((-.457, 0, .457)) # positioning of vertical lines
    horizontal_lines: np.ndarray = np.array((back_line_position, tee_line_position, hog_line_position, pitch_length / 2, 33.832, 40.233, 42.062)) # positioning of horizontal lines
    num_stones_per_end: int = 16
    def __init__(self, starting_color: Optional[StoneColor] = None):
        self.reset(starting_color)

    def reset(self, starting_color: Optional[StoneColor] = None):
        self.stones: List[Stone] = []
        self.next_stone_colour = starting_color or np.random.choice([StoneColor.RED, StoneColor.YELLOW])

    def step(self, simulation_constants: SimulationConstants = SimulationConstants()) -> SimulationState:
        finished = SimulationState.FINISHED
        update_accuracy = False
        invalid_stone_indices = []
        for i, stone in enumerate(self.stones):
            if stone.step(simulation_constants) == SimulationState.UNFINISHED:
                finished = SimulationState.UNFINISHED
                if -stone.position[1] < self.hog_line_position:
                    update_accuracy = True
            if self.out_of_bounds(stone):
                invalid_stone_indices.append(i)
        for invalid_index in reversed(invalid_stone_indices):
            self.stones.pop(invalid_index)
        Stone.handle_collisions(self.stones, constants=simulation_constants)
        if update_accuracy:
            simulation_constants.accuracy = max(simulation_constants.accuracy, Accuracy.MID)
        return finished

    def render(self) -> Canvas:
        canvas = Canvas(self, pixels_per_meter=920//(self.pitch_length / 2))
        canvas.draw_vertical_lines(self.vertical_lines)
        canvas.draw_targets(buffer=self.tee_line_position, radii=self.target_radii)
        canvas.draw_horizontal_lines(self.horizontal_lines)

        for stone in self.stones:
            canvas.draw_stone(stone)
        return canvas

    def out_of_bounds(self, stone: Stone) -> bool:
        return np.abs(stone.position[0]) > self.pitch_width / 2 or stone.position[1] > -self.back_line_position + stone.outer_radius

    def button_distance(self, stone: Stone) -> float:
        return np.linalg.norm(stone.position - self.button_position)

    def in_house(self, stone: Stone) -> bool:
        return self.button_distance(stone) < self.house_radius

    def in_fgz(self, stone: Stone) -> bool:
        """determines if a stone is in the free guard zone"""
        return (-stone.position[1] >= self.tee_line_position and -stone.position[1] <= self.hog_line_position) and not self.in_house(stone)

    def display(self, constants: SimulationConstants = SimulationConstants()):
        self.render().display(constants)

    def create_stone(self, stone_throw: StoneThrow):
        return Stone(
            color=stone_throw.color,
            velocity=stone_throw.velocity,
            angle=stone_throw.angle,
            spin=stone_throw.spin,
            position=(0, self.button_position[1]-self.starting_button_distance),
            curling_constants=self.physical_constants
        )

    def throw(self, stone_throw: StoneThrow, constants: SimulationConstants = SimulationConstants(), display: bool=False):
        assert stone_throw.color == self.next_stone_colour
        self.next_stone_colour = ~self.next_stone_colour
        self.stones.append(
            self.create_stone(
                stone_throw
            )

        )
        constants.reset()
        while self.step(constants) == SimulationState.UNFINISHED:
            if display:
                self.display(constants)

    def evaluate_position(self):
        stone_distances = [self.button_distance(stone) for stone in self.stones]
        if len(self.stones) == 0:
            distance_ordering = []
        else:
            distance_ordering = np.argsort(stone_distances)
        ordered_stones = [self.stones[index] for index in distance_ordering]
        score = 0
        for stone in ordered_stones:
            if score * stone.color < 0 or not self.in_house(stone):
                break
            score += stone.color
        return score

class Canvas:
    WINDOW_NAME = "Curling"
    DISPLAY_TIME = DisplayTime.TWICE_SPEED
    def __init__(self, curling: Curling, pixels_per_meter: int = 20):
        self.pitch_width = curling.pitch_width
        self.pitch_length = curling.pitch_length / 2
        self.pixels_per_meter = pixels_per_meter
        self.canvas_width = int(self.pitch_width * pixels_per_meter)
        self.canvas_height = int(self.pitch_length * pixels_per_meter)
        self._canvas = np.tile(np.array(Colors.BACKGROUND.value).astype(np.uint8), (self.canvas_height, self.canvas_width, 1))

    def adjust_coordinates(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        return (int((xy[0] + self.pitch_width / 2) * self.pixels_per_meter), int((-xy[1]) * self.pixels_per_meter))

    def convert_radius(self, radius: float) -> float:
        return int(radius * self.pixels_per_meter)

    def draw_target(self, radii: List[float], offset: float):
        TARGET_COLOURS = (Colors.RED, Colors.WHITE, Colors.BLUE, Colors.WHITE)
        for color, radius in zip(TARGET_COLOURS, reversed(sorted((radii)))):
            cv2.circle(self._canvas, center=self.adjust_coordinates((0, offset)), radius=self.convert_radius(radius), color=color.value, thickness=-1)

    def draw_horizontal_lines(self, lines: List[float]):
        for height in lines:
            cv2.line(self._canvas, self.adjust_coordinates((-self.pitch_width, -height)), self.adjust_coordinates((self.pitch_width, -height)), color=Colors.WHITE.value, thickness=1)

    def draw_vertical_lines(self, lines: List[float]):
        for width in lines:
            cv2.line(self._canvas, self.adjust_coordinates((width, 0)), self.adjust_coordinates((width, -self.pitch_length)), color=Colors.WHITE.value, thickness=1)

    def draw_targets(self, radii: List[float], buffer: float):
        self.draw_target(radii=radii, offset=-buffer)

    def draw_stone(self, stone: Stone):
        stone_color = Colors.RED if stone.color == StoneColor.RED else Colors.YELLOW
        cv2.circle(self._canvas, center=self.adjust_coordinates(stone.position), radius=self.convert_radius(stone.outer_radius), color=stone_color.value, thickness=-1)
        cv2.circle(self._canvas, center=self.adjust_coordinates(stone.position), radius=self.convert_radius(stone.outer_radius), color=Colors.GRAY.value, thickness=1)
        handle_offset = stone.outer_radius * (np.cos(stone.angular_position), np.sin(stone.angular_position))
        cv2.line(self._canvas, pt1=self.adjust_coordinates(stone.position + handle_offset), pt2=self.adjust_coordinates(stone.position - handle_offset), color=Colors.GRAY.value, thickness=1)

    def get_canvas(self)-> np.ndarray:
        return self._canvas

    def display(self, constants: SimulationConstants = SimulationConstants()):
        cv2.imshow(self.WINDOW_NAME, self._canvas)
        linear_transform = self.DISPLAY_TIME.value
        cv2.waitKey(int(linear_transform(1000 * constants.dt)))

@dataclass
class StoneThrow:
    bounds: ClassVar[np.ndarray] = np.array([
        (1.3, 2.),
        (-.1, .1),
        (-4, 4)
    ]).astype(float)
    random_parameters: ClassVar[np.ndarray] = np.array([
        (1.41, .05),
        (0., .04),
        (0., 1.),
    ])
    color: StoneColor
    sqrt_velocity: float
    angle: float
    spin: float
    def __post_init__(self):
        self.velocity = self.sqrt_velocity ** 2