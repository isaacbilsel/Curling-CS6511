import numpy as np

from typing import List, Tuple

from .constants import Accuracy, PhysicalConstants, SimulationConstants
from .enums import StoneColor, SimulationState


class Stone:
    mass: np.floating = np.array(19.0)
    height: np.floating = np.array(0.1143)
    ring_radius: np.floating = np.array(0.065)
    outer_radius: np.floating = np.array(0.142)
    angular_acceleration: np.floating = np.array(0.0)
    angular_velocity: np.floating = np.array(1.5)
    angular_position: np.floating = np.array(0.0)
    weight: np.floating = np.array(mass * PhysicalConstants.g)
    moment_of_inertia: np.floating = np.array(0.5 * mass * outer_radius ** 2)
    coefficient_of_restitution: np.floating = np.array(0.5)
    coefficient_of_friction: np.floating = np.array(0.2)
    acceleration = np.array((0.0, 0.0))
    velocity = np.array((0.01, 2.2))

    def __init__(
        self,
        color: StoneColor,
        position: Tuple[float, float] = (0, 0),
        velocity: float = 0,
        angle: float = 0,
        spin: float = 0,
        curling_constants: PhysicalConstants = PhysicalConstants,
    ):
        self.color = color
        self.curling_constants = curling_constants
        self.position = np.array(position)
        self.velocity = np.array((-np.sin(angle), np.cos(angle))) * velocity
        self.angular_position = np.array(0.0)
        self.angular_velocity = np.array(spin, dtype=float)

    def step(self, simulation_constants: SimulationConstants = SimulationConstants()) -> SimulationState:
        if np.linalg.norm(self.velocity) < simulation_constants.eps:
            self.angular_velocity = 0.0
            return SimulationState.FINISHED

        dt = simulation_constants.dt
        theta = np.arange(0, 2 * np.pi, simulation_constants.dtheta)
        relative_point_position = np.array((-np.sin(theta), np.cos(theta))).T * self.ring_radius
        normalised_tangent = np.array((-np.cos(theta), -np.sin(theta))).T
        phi = theta - np.arctan2(-self.velocity[0], self.velocity[1])

        point_velocity = self.angular_velocity * self.ring_radius * normalised_tangent + self.velocity
        point_speed = np.linalg.norm(point_velocity, axis=-1)

        normal_force = self.weight / simulation_constants.num_points_on_circle
        forward_ratio = np.cos(phi)
        mu = self.curling_constants.calculate_friction(point_speed, forward_ratio)
        frictional_force = np.minimum(
            normal_force * mu,
            point_speed * (self.mass / simulation_constants.num_points_on_circle) / dt,
        )
        frictional_force = np.tile(frictional_force, (2, 1)).T * -point_velocity / (np.tile(point_speed, (2, 1))).T

        net_force = frictional_force.sum(axis=0)
        torque = -np.linalg.det(np.stack((frictional_force, relative_point_position), axis=1)).sum(axis=-1)

        self.angular_acceleration = torque / self.moment_of_inertia
        self.angular_velocity += self.angular_acceleration * dt
        self.angular_position += self.angular_velocity * dt
        self.acceleration = net_force / self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        return SimulationState.UNFINISHED

    def unstep(self, simulation_constants: SimulationConstants = SimulationConstants()) -> SimulationState:
        dt = simulation_constants.dt
        self.position -= self.velocity * dt
        self.velocity -= self.acceleration * dt
        self.angular_position -= self.angular_velocity * dt
        self.angular_velocity -= self.angular_acceleration * dt
        return SimulationState.UNFINISHED

    @staticmethod
    def handle_collisions(stones: List["Stone"], constants: SimulationConstants = SimulationConstants()):
        impulses = np.zeros((len(stones), 2))
        torques = np.zeros((len(stones),))
        for i in range(len(stones)):
            stone1 = stones[i]
            for j in range(i):
                stone2 = stones[j]
                normal_vector = stone1.position - stone2.position
                distance = np.linalg.norm(normal_vector)
                if distance <= stone1.outer_radius + stone2.outer_radius:
                    normal_vector /= distance
                    tangent_vector = np.array((-normal_vector[1], normal_vector[0]))
                    relative_velocity = stone1.velocity - stone2.velocity
                    relative_normal_velocity = np.dot(relative_velocity, normal_vector)
                    relative_tangent_velocity = np.dot(relative_velocity, tangent_vector)
                    relative_tangent_velocity -= stone1.angular_velocity * stone1.outer_radius + stone2.angular_velocity * stone2.outer_radius

                    impulse = -(1 + Stone.coefficient_of_restitution) * relative_normal_velocity / (1 / stone1.mass + 1 / stone2.mass)
                    impulse *= normal_vector
                    impulses[i] += impulse
                    impulses[j] -= impulse

                    tangent_impulse = min(
                        Stone.coefficient_of_friction * np.linalg.norm(impulse),
                        (1 + Stone.coefficient_of_restitution) * np.abs(relative_tangent_velocity)
                        / (
                            stone1.outer_radius ** 2 / stone1.moment_of_inertia
                            + 1 / stone1.mass
                            + stone2.outer_radius ** 2 / stone2.moment_of_inertia
                            + 1 / stone2.mass
                        ),
                    ) * np.sign(relative_tangent_velocity)

                    torques[i] += tangent_impulse * stone1.outer_radius
                    torques[j] -= tangent_impulse * stone2.outer_radius

                    tangent_impulse *= -tangent_vector
                    impulses[i] += tangent_impulse
                    impulses[j] -= tangent_impulse

        collision = ((torques != 0).any() or (impulses != 0).any())
        if collision and constants.accuracy != Accuracy.HIGH:
            for stone in stones:
                stone.unstep(constants)
            constants.accuracy = Accuracy.HIGH
            return

        dt = constants.dt
        for stone, impulse, torque in zip(stones, impulses, torques):
            if (torque != 0).any() or (impulse != 0).any():
                stone.unstep(constants)
                stone.velocity += impulse / stone.mass
                stone.position += stone.velocity * dt
                stone.angular_velocity += torque / stone.moment_of_inertia

        touching = True
        while touching:
            touching = False
            for i in range(len(stones)):
                stone1 = stones[i]
                for j in range(i):
                    stone2 = stones[j]
                    normal_vector = stone1.position - stone2.position
                    distance = np.linalg.norm(normal_vector)
                    if distance <= stone1.outer_radius + stone2.outer_radius:
                        touching = True
                        nudge_distance = (stone1.outer_radius + stone2.outer_radius - distance) / 2 + constants.eps
                        stone1.position += normal_vector * nudge_distance
                        stone2.position -= normal_vector * nudge_distance

        if collision:
            constants.accuracy = Accuracy.MID
