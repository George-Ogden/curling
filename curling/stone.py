import jax.numpy as jnp
import numpy as np

from typing import List, Tuple

from .constants import Accuracy, PhysicalConstants, SimulationConstants
from .enums import StoneColor, SimulationState

class Stone:
    mass: jnp.floating = jnp.array(19.) # stones between 17.24-19.96
    height: jnp.floating = jnp.array(0.1143) # height of the stone
    ring_radius: jnp.floating = jnp.array(0.065) # radius of the inner ring
    outer_radius: jnp.floating = jnp.array(0.142) # radius of the entire stone
    angular_acceleration: jnp.floating = jnp.array(0.) # angular acceleration
    angular_velocity: jnp.floating = jnp.array(1.5) # clockwise is negative (rad/s)
    angular_position: jnp.floating = jnp.array(0.)
    weight: jnp.floating = jnp.array(mass * PhysicalConstants.g)
    moment_of_inertia: jnp.floating = jnp.array(.5 * mass * outer_radius ** 2) # I = 1/2 mr^2
    coefficient_of_restitution: jnp.floating = jnp.array(.5) # coefficient of restitution between stones
    coefficient_of_friction: jnp.floating = jnp.array(.2) # coefficient of friction between stones
    acceleration = jnp.array((0., 0.)) # xy acceleration of the stone
    velocity = jnp.array((0.01, 2.2)) # xy velocity of the stone
    def __init__(self, color: StoneColor, position: Tuple[float, float] = (0, 0), velocity: float = 0, angle: float = 0, spin: float = 0, curling_constants: PhysicalConstants = PhysicalConstants):
        """create a moving stone (equivalent to throwing)

        Args:
            x_position (float): position of the thrower from the centre line (-2, 2)
            velocity (float): velocity of the stone when it is released (?)
            angle (float): angle between the centre line and direction of the stone in radians ()
            spin (float): amount of spin on the stone in radians (-2, 2)
        """
        self.color = color
        self.curling_constants = curling_constants
        self.position = jnp.array(position)
        self.velocity = jnp.array((-jnp.sin(angle), jnp.cos(angle))) * velocity
        self.angular_position = jnp.array(0.)
        self.angular_velocity = jnp.array(spin, dtype=float)

    def step(self, simulation_constants: SimulationConstants=SimulationConstants()) -> SimulationState:
        if jnp.linalg.norm(self.velocity) < simulation_constants.eps:
            self.angular_velocity = 0.
            return SimulationState.FINISHED

        dt = simulation_constants.dt
        theta = jnp.arange(0, 2 * jnp.pi, simulation_constants.dtheta)
        relative_point_position = jnp.array((-jnp.sin(theta), jnp.cos(theta))).T * self.ring_radius # position of this point relative to centre of stone
        normalised_tangent = jnp.array((-jnp.cos(theta), -jnp.sin(theta))).T # direction of the tangent normal to the radius from the centre
        phi = theta - jnp.arctan2(-self.velocity[0], self.velocity[1]) # angle relative to direction of motion

        point_velocity = self.angular_velocity * self.ring_radius * normalised_tangent + self.velocity # speed relative to the ground
        point_speed = jnp.linalg.norm(point_velocity, axis=-1)

        # divide weight between all the points
        normal_force = self.weight / simulation_constants.num_points_on_circle
        forward_ratio = jnp.cos(phi) # ratio along the disc used to calculate friction
        mu = self.curling_constants.calculate_friction(point_speed, forward_ratio)
        frictional_force = jnp.minimum(normal_force * mu, point_speed * (self.mass / simulation_constants.num_points_on_circle) / dt) # F <= mu * N
        # point force in opposite direction to velocity
        frictional_force = jnp.tile(frictional_force, (2, 1)).T * -point_velocity / (jnp.tile(point_speed, (2, 1))).T

        # add the frictional force
        net_force = frictional_force.sum(axis=0)
        # torque magnitude
        torque = -jnp.linalg.det(jnp.stack((frictional_force, relative_point_position), axis=1)).sum(axis=-1)

        # update values
        self.angular_acceleration = torque / self.moment_of_inertia
        self.angular_velocity = self.angular_velocity + self.angular_acceleration * dt
        self.angular_position = self.angular_position + self.angular_velocity * dt
        self.acceleration = net_force / self.mass
        self.velocity = self.velocity + self.acceleration * dt
        self.position = self.position + self.velocity * dt
        return SimulationState.UNFINISHED
    
    def unstep(self, simulation_constants: SimulationConstants = SimulationConstants()) -> SimulationState:
        dt = simulation_constants.dt
        # revert a step
        self.position = self.position - self.velocity * dt
        self.velocity = self.velocity - self.acceleration * dt
        self.angular_position = self.angular_position - self.angular_velocity * dt
        self.angular_velocity = self.angular_velocity - self.angular_acceleration * dt

        # unstep must always happen even if stone is stopping
        return SimulationState.UNFINISHED

    @staticmethod
    def handle_collisions(stones: List["Stone"], constants: SimulationConstants = SimulationConstants()):
        impulses = jnp.zeros((len(stones), 2))
        torques = jnp.zeros((len(stones), ))
        # could be rewritten more efficiently if stones are sorted by y coordinate and then values are recalculated
        for i in range(len(stones)):
            stone1 = stones[i]
            for j in range(i):
                stone2 = stones[j]
                normal_vector = stone1.position - stone2.position
                distance = jnp.linalg.norm(normal_vector)
                if distance <= stone1.outer_radius + stone2.outer_radius:
                    normal_vector /= distance
                    tangent_vector = jnp.array((-normal_vector[1], normal_vector[0]))
                    relative_velocity = stone1.velocity - stone2.velocity
                    relative_normal_velocity = jnp.dot(relative_velocity, normal_vector) # relative velocity in normal direction
                    relative_tangent_velocity = jnp.dot(relative_velocity, tangent_vector) # relative velocity in the tangent direction
                    relative_tangent_velocity = relative_tangent_velocity - stone1.angular_velocity * stone1.outer_radius + stone2.angular_velocity * stone2.outer_radius

                    impulse = -(1 + Stone.coefficient_of_restitution) * relative_normal_velocity / (1 / stone1.mass + 1 / stone2.mass)
                    impulse = impulse *normal_vector
                    impulses = impulses.at[i].add(impulse)
                    impulses = impulses.at[j].add(-impulse)

                    # tangent impulse is limited by the relative tangent velocity
                    tangent_impulse = min(
                        Stone.coefficient_of_friction * jnp.linalg.norm(impulse),
                        (1 + Stone.coefficient_of_restitution) * jnp.abs(relative_tangent_velocity) / (stone1.outer_radius ** 2 / stone1.moment_of_inertia + 1 / stone1.mass + stone2.outer_radius ** 2 / stone2.moment_of_inertia + 1 / stone2.mass)
                    ) * jnp.sign(relative_tangent_velocity)

                    torques = torques.at[i].add(tangent_impulse * stone1.outer_radius)
                    torques = torques.at[j].add(-tangent_impulse * stone2.outer_radius)

                    tangent_impulse = tangent_impulse * -tangent_vector
                    impulses = impulses.at[i].add(tangent_impulse)
                    impulses = impulses.at[j].add(-tangent_impulse)

        collision = ((torques != 0).any() or (impulses != 0).any())
        if collision and constants.accuracy != Accuracy.HIGH:
            # require high accuracy for collision simulations
            for stone in stones:
                stone.unstep(constants)
            constants.accuracy = Accuracy.HIGH
            return

        dt = constants.dt
        for stone, impulse, torque in zip(stones, impulses, torques):
            if (torque != 0).any() or (impulse != 0).any():
                stone.unstep(constants)

                # update to next state
                stone.velocity = stone.velocity + impulse / stone.mass
                stone.position = stone.position + stone.velocity * dt
                stone.angular_velocity = stone.angular_velocity + torque / stone.moment_of_inertia

        # make sure no stones are still touching
        touching = True
        while touching:
            touching = False
            for i in range(len(stones)):
                stone1 = stones[i]
                for j in range(i):
                    stone2 = stones[j]
                    normal_vector = stone1.position - stone2.position
                    distance = jnp.linalg.norm(normal_vector)
                    if distance <= stone1.outer_radius + stone2.outer_radius:
                        touching = True
                        nudge_distance = (stone1.outer_radius + stone2.outer_radius - distance) / 2 + constants.eps
                        stone1.position = stone1.position + normal_vector * nudge_distance
                        stone2.position = stone2.position - normal_vector * nudge_distance

        if collision:
            # reduce simulation time 
            # high accuracy only needed for the collisions themselves
            constants.accuracy = Accuracy.MID