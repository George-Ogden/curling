import jax.numpy as jnp

import pytest

from curling import Curling, Stone, StoneColor, StoneThrow, SimulationConstants

approx_constants = SimulationConstants(time_intervals=.1, num_points_on_circle=10)
accurate_constants = SimulationConstants(time_intervals=.02)

def test_red_initialisation():
    curling = Curling(StoneColor.RED)
    assert len(curling.stones) == 0
    assert curling.next_stone_colour == StoneColor.RED

def test_yellow_initialisation():
    curling = Curling(StoneColor.YELLOW)
    assert len(curling.stones) == 0
    assert curling.next_stone_colour == StoneColor.YELLOW

def test_colour_changes():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        spin=0,
        angle=0,
        sqrt_velocity=1.41
    ), constants=approx_constants)
    with pytest.raises(AssertionError) as e:
        curling.throw(StoneThrow(
            StoneColor.RED,
            spin=0,
            angle=0,
            sqrt_velocity=1.41
        ), constants=approx_constants)

def test_evaluate_with_single_stone():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            sqrt_velocity=1.41,
            angle=0,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.RED

    curling.reset(StoneColor.YELLOW)
    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            sqrt_velocity=1.41,
            angle=0,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.YELLOW

def test_evaluate_with_double_stone():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            sqrt_velocity=1.41,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            sqrt_velocity=1.41,
            angle=0.05,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.RED,
            sqrt_velocity=1.41,
            angle=0.02,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.RED * 2

def test_evaluate_after_collision():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            sqrt_velocity=1.41,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            sqrt_velocity=1.5,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.RED,
            sqrt_velocity=2,
            angle=0.5,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            sqrt_velocity=1.41,
            angle=0,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.YELLOW * 2

def test_evaluate_with_split_stones():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            sqrt_velocity=1.41,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            sqrt_velocity=1.41,
            angle=0.02,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.RED,
            sqrt_velocity=1.41,
            angle=0.05,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.RED

def test_free_guard_zone():
    curling = Curling()
    stone = Stone(color=StoneColor.RED, position=(0,0))
    assert not curling.in_fgz(stone)

    stone.position = jnp.array((0, -curling.tee_line_position + curling.target_radii[0]))
    assert not curling.in_fgz(stone)

    stone.position = jnp.array((1, -curling.tee_line_position - curling.target_radii[0]))
    assert not curling.in_fgz(stone)

    stone.position = jnp.array((0, -curling.tee_line_position - curling.target_radii[-1]))
    assert not curling.in_fgz(stone)

    stone.position = jnp.array((0, -curling.tee_line_position - curling.target_radii[-1] * 2))
    assert curling.in_fgz(stone)

    stone.position = jnp.array((-1, -curling.hog_line_position + curling.target_radii[0]))
    assert curling.in_fgz(stone)

    stone.position = jnp.array((0, -curling.hog_line_position - curling.target_radii[0]))
    assert not curling.in_fgz(stone)

    stone.position = jnp.array((1, -curling.horizontal_lines[-2]))
    assert not curling.in_fgz(stone)

def test_in_house():
    curling = Curling()
    stone = Stone(color=StoneColor.RED, position=(0,0))
    assert not curling.in_house(stone)

    stone.position = jnp.array((0, -curling.tee_line_position + curling.target_radii[0]))
    assert curling.in_house(stone)

    stone.position = jnp.array((1, -curling.tee_line_position - curling.target_radii[0]))
    assert curling.in_house(stone)

    stone.position = jnp.array((curling.target_radii[0], -curling.tee_line_position + curling.target_radii[-2]))
    assert curling.in_house(stone)

    stone.position = jnp.array((-1, -curling.tee_line_position - curling.target_radii[-1]))
    assert not curling.in_house(stone)

    stone.position = jnp.array((-curling.target_radii[-1], -curling.tee_line_position))
    assert curling.in_house(stone)

    stone.position = jnp.array((-curling.hog_line_position))
    assert not curling.in_house(stone)

def test_in_house_scoring():
    curling = Curling()
    curling.stones.append(Stone(color=StoneColor.RED, position=(0, -curling.tee_line_position -curling.target_radii[-1])))
    curling.stones.append(Stone(color=StoneColor.RED, position=(0, -curling.tee_line_position -curling.target_radii[0])))
    assert curling.evaluate_position() == StoneColor.RED * 2

def test_out_of_house_scoring():
    curling = Curling()
    curling.stones.append(Stone(color=StoneColor.RED, position=(0, -curling.hog_line_position)))
    assert curling.evaluate_position() == 0

def test_mixed_house_scoring():
    curling = Curling()
    curling.stones.append(Stone(color=StoneColor.RED, position=(0, -curling.tee_line_position -curling.target_radii[-1])))
    curling.stones.append(Stone(color=StoneColor.RED, position=(0, -curling.hog_line_position)))
    assert curling.evaluate_position() == StoneColor.RED

def test_bounds():
    curling = Curling()
    assert curling.out_of_bounds(Stone(color=StoneColor.RED, position=(0, .1)))
    assert curling.out_of_bounds(Stone(color=StoneColor.RED, position=(0, 0.)))
    assert curling.out_of_bounds(Stone(color=StoneColor.RED, position=(0, -curling.back_line_position / 2)))
    assert curling.out_of_bounds(Stone(color=StoneColor.RED, position=(curling.pitch_width + .1, 0)))
    assert curling.out_of_bounds(Stone(color=StoneColor.RED, position=(-curling.pitch_width - .1, 0)))
    
    assert not curling.out_of_bounds(Stone(color=StoneColor.RED, position=(0, -curling.pitch_length / 2)))
    assert not curling.out_of_bounds(Stone(color=StoneColor.RED, position=(0, -curling.hog_line_position)))
    assert not curling.out_of_bounds(Stone(color=StoneColor.RED, position=(0, -curling.tee_line_position)))
    assert not curling.out_of_bounds(Stone(color=StoneColor.RED, position=(0, -curling.back_line_position)))

    assert not curling.out_of_bounds(Stone(color=StoneColor.RED, position=(curling.pitch_width * .4, -curling.hog_line_position)))
    assert not curling.out_of_bounds(Stone(color=StoneColor.RED, position=(-curling.pitch_width * .4, -curling.hog_line_position)))