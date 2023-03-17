import numpy as np
import cv2

from pytest import mark

from curling import Curling, SimulationConstants, StoneColor, StoneThrow
from curling.enums import Colors, DisplayTime, LinearTransform
from curling.curling import Canvas

approx_constants = SimulationConstants(time_intervals=(.5, .1), num_points_on_circle=10)
accurate_constants = SimulationConstants(time_intervals=(.1, .02))

@mark.display
def test_display():
    curling = Curling(StoneColor.YELLOW)
    curling.throw(StoneThrow(
        StoneColor.YELLOW,
        angle=.01,
        velocity=2,
        spin=0
    ), constants=approx_constants)
    curling.display()
    assert cv2.getWindowProperty(Canvas.WINDOW_NAME, cv2.WND_PROP_VISIBLE) != -1

    # cleanup
    cv2.destroyAllWindows()

@mark.display
def test_image():
    curling = Curling(StoneColor.YELLOW)
    curling.throw(StoneThrow(
        StoneColor.YELLOW,
        angle=.01,
        velocity=1.4,
        spin=0
    ), constants=approx_constants)
    image = curling.render().get_canvas()
    assert np.abs(image.shape[0] / image.shape[1] - (curling.pitch_length / 2) / curling.pitch_width) < 0.1
    assert (image == Colors.YELLOW.value).any()
    assert (image == Colors.BACKGROUND.value).sum() > np.prod(image.shape[:-1]) / 2

def test_linear_transform():
    transform = LinearTransform(10, 5)
    assert transform(6) == 65

def test_display_times():
    for time in DisplayTime:
        if time == DisplayTime.FOREVER:
            assert int(time.value(1000 * accurate_constants.dt)) == 0
        else:
            assert int(time.value(1000 * accurate_constants.dt)) > 0

def test_horizontal_lines_are_symmetric():
    line_sums = Curling.horizontal_lines + Curling.horizontal_lines[::-1]
    assert (line_sums == Curling.pitch_length).all()

def test_vertical_lines_are_symmetric():
    line_sums = Curling.vertical_lines + Curling.vertical_lines[::-1]
    assert (line_sums == 0).all()