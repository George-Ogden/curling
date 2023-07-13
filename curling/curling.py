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
    num_stones_per_end: int = 16 # 8 red stones and 8 yellow stones
    def __init__(self, starting_color: Optional[StoneColor] = None):
        """initialise curling game

        Args:
            starting_color (Optional[StoneColor], optional): color of starting player (or None for random). Defaults to None.
        """
        self.reset(starting_color)

    def reset(self, starting_color: Optional[StoneColor] = None):
        """reset curling game

        Args:
            starting_color (Optional[StoneColor], optional): color of starting player (or none for random). Defaults to None.
        """
        self.stones: List[Stone] = []
        self.next_stone_color = starting_color or np.random.choice([StoneColor.RED, StoneColor.YELLOW])

    def step(self, simulation_constants: SimulationConstants = SimulationConstants()) -> SimulationState:
        """step the simulation one timestep (simulation_constants.dt)

        Args:
            simulation_constants (SimulationConstants, optional): specify a set of constants to use for the simulation step. Defaults to SimulationConstants().

        Returns:
            SimulationState: whether the simulation is finished or not
        """
        finished = SimulationState.FINISHED
        update_accuracy = False
        invalid_stone_indices = [] # store stones that are invalid
        for i, stone in enumerate(self.stones):
            if stone.step(simulation_constants) == SimulationState.UNFINISHED:
                # if any stone is still moving, the simulation is not finished
                finished = SimulationState.UNFINISHED
                # increase accuracy if a stone is past the hog line
                if -stone.position[1] < self.hog_line_position:
                    update_accuracy = True
            if self.out_of_bounds(stone):
                invalid_stone_indices.append(i)

        # remove invalid stones from further processing
        for invalid_index in reversed(invalid_stone_indices):
            self.stones.pop(invalid_index)
        # handle collisions
        Stone.handle_collisions(self.stones, constants=simulation_constants)
        if update_accuracy:
            # increase the accuracy to at least MID
            simulation_constants.accuracy = max(simulation_constants.accuracy, Accuracy.MID)
        return finished

    def render(self) -> Canvas:
        """create a canvas and render the curling game

        Returns:
            Canvas: canvas with rendered curling game
        """
        # create canvas with height of 920 pixels
        canvas = Canvas(self, pixels_per_meter=920//(self.pitch_length / 2))

        # render graphics
        canvas.draw_vertical_lines(self.vertical_lines)
        canvas.draw_targets(buffer=self.tee_line_position, radii=self.target_radii)
        canvas.draw_horizontal_lines(self.horizontal_lines)

        for stone in self.stones:
            canvas.draw_stone(stone)
        return canvas

    def out_of_bounds(self, stone: Stone) -> bool:
        """check if a stone is out of bounds"""
        # check if stone is wider than the pitch width (half on either side)
        # check if the stone has completely passed the back line
        return np.abs(stone.position[0]) > self.pitch_width / 2 or stone.position[1] > -self.back_line_position + stone.outer_radius

    def button_distance(self, stone: Stone) -> float:
        """calculate the distance from the button to a stone"""
        return np.linalg.norm(stone.position - self.button_position)

    def in_house(self, stone: Stone) -> bool:
        """determines if a stone is in the house"""
        return self.button_distance(stone) < self.house_radius

    def in_fgz(self, stone: Stone) -> bool:
        """determines if a stone is in the free guard zone"""
        return (-stone.position[1] >= self.tee_line_position and -stone.position[1] <= self.hog_line_position) and not self.in_house(stone)

    def display(self, constants: SimulationConstants = SimulationConstants()):
        """display the curling game
        
        Args:
            constants (SimulationConstants, optional): specify a set of constants to use for the simulation step. Defaults to SimulationConstants().            
        """
        self.render().display(constants)

    def create_stone(self, stone_throw: StoneThrow) -> Stone:
        """create a stone from a stone throw"""
        return Stone(
            color=stone_throw.color,
            velocity=stone_throw.velocity,
            angle=stone_throw.angle,
            spin=stone_throw.spin,
            position=(0, self.button_position[1]-self.starting_button_distance),
            curling_constants=self.physical_constants
        )

    def throw(self, stone_throw: StoneThrow, constants: SimulationConstants = SimulationConstants(), display: bool=False):
        """update the curling game with a stone throw

        Args:
            stone_throw (StoneThrow): stone throw of current player
            constants (SimulationConstants, optional): specify a set of constants to use for the simulation. Defaults to SimulationConstants().
            display (bool, optional): whether to display the throw. Defaults to False.
        """
        assert stone_throw.color == self.next_stone_color, f"It is {self.next_stone_color.name}'s turn, not {stone_throw.color.name}'s turn"
        # change the next stone color
        self.next_stone_color = ~self.next_stone_color
        # add a new moving stone to the game
        self.stones.append(
            self.create_stone(
                stone_throw
            )

        )
        constants.reset()
        # run the simulation until it is finished
        while self.step(constants) == SimulationState.UNFINISHED:
            if display:
                self.display(constants)

    def evaluate_position(self) -> int:
        """evaluate the current position of the stones
        the score is n * color, where n is the number of stones of the color closest to the button
        a score of 0 means all stones are out of the house

        Returns:
            int: score of the current position
        """
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
    """canvas for rendering the curling game"""
    WINDOW_NAME = "Curling"
    DISPLAY_TIME = DisplayTime.TWICE_SPEED
    def __init__(self, curling: Curling, pixels_per_meter: int = 20):
        """create a canvas for rendering the curling game

        Args:
            curling (Curling): curling game to render
            pixels_per_meter (int, optional): pixels per meter. Defaults to 20.
        """
        self.pitch_width = curling.pitch_width
        self.pitch_length = curling.pitch_length / 2
        self.pixels_per_meter = pixels_per_meter
        self.canvas_width = int(self.pitch_width * pixels_per_meter)
        self.canvas_height = int(self.pitch_length * pixels_per_meter)
        self._canvas = np.tile(np.array(Colors.BACKGROUND.value).astype(np.uint8), (self.canvas_height, self.canvas_width, 1))

    def adjust_coordinates(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """adjust real world coordinates to canvas coordinates"""
        return (int((xy[0] + self.pitch_width / 2) * self.pixels_per_meter), int((-xy[1]) * self.pixels_per_meter))

    def convert_radius(self, radius: float) -> float:
        """convert a radius in meters to a radius in pixels"""
        return int(radius * self.pixels_per_meter)

    def draw_target(self, radii: List[float], offset: float):
        """draw the target on the canvas"""
        TARGET_colorS = (Colors.RED, Colors.WHITE, Colors.BLUE, Colors.WHITE)
        for color, radius in zip(TARGET_colorS, reversed(sorted((radii)))):
            cv2.circle(self._canvas, center=self.adjust_coordinates((0, offset)), radius=self.convert_radius(radius), color=color.value, thickness=-1)

    def draw_horizontal_lines(self, lines: List[float]):
        """draw horizontal lines on the canvas"""
        for height in lines:
            cv2.line(self._canvas, self.adjust_coordinates((-self.pitch_width, -height)), self.adjust_coordinates((self.pitch_width, -height)), color=Colors.WHITE.value, thickness=1)

    def draw_vertical_lines(self, lines: List[float]):
        """draw vertical lines on the canvas"""
        for width in lines:
            cv2.line(self._canvas, self.adjust_coordinates((width, 0)), self.adjust_coordinates((width, -self.pitch_length)), color=Colors.WHITE.value, thickness=1)

    def draw_targets(self, radii: List[float], buffer: float):
        """draw the targets on the canvas"""
        self.draw_target(radii=radii, offset=-buffer)

    def draw_stone(self, stone: Stone):
        """draw a stone on the canvas"""
        stone_color = Colors.RED if stone.color == StoneColor.RED else Colors.YELLOW
        # band of color
        cv2.circle(self._canvas, center=self.adjust_coordinates(stone.position), radius=self.convert_radius(stone.outer_radius), color=stone_color.value, thickness=-1)
        # grey centre
        cv2.circle(self._canvas, center=self.adjust_coordinates(stone.position), radius=self.convert_radius(stone.outer_radius), color=Colors.GRAY.value, thickness=1)
        handle_offset = stone.outer_radius * (np.cos(stone.angular_position), np.sin(stone.angular_position))
        # handle
        cv2.line(self._canvas, pt1=self.adjust_coordinates(stone.position + handle_offset), pt2=self.adjust_coordinates(stone.position - handle_offset), color=Colors.GRAY.value, thickness=1)

    def get_canvas(self)-> np.ndarray:
        """get the canvas"""
        return self._canvas

    def display(self, constants: SimulationConstants = SimulationConstants()):
        """display the canvas
        
        Args:
            constants (SimulationConstants, optional): simulation constants to determine speed of animation. Defaults to SimulationConstants().
        """
        cv2.imshow(self.WINDOW_NAME, self._canvas)
        linear_transform = self.DISPLAY_TIME.value
        cv2.waitKey(int(linear_transform(1000 * constants.dt)))

@dataclass
class StoneThrow:
    """class for storing the parameters of a stone throw"""
    bounds: ClassVar[np.ndarray] = np.array([
        (1.3, 2.),
        (-.1, .1),
        (-4, 4)
    ]).astype(float)
    """random parameters are drawn from a normal distribution with (mean, std) given by bounds"""
    random_parameters: ClassVar[np.ndarray] = np.array([
        (1.41, .03),
        (0., .04),
        (0., 1.),
    ])
    color: StoneColor
    sqrt_velocity: float
    angle: float
    spin: float
    def __post_init__(self):
        self.velocity = self.sqrt_velocity ** 2
