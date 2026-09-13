"""Compile and step the physical NeuroMechFly body using FlyGym 2.1/MuJoCo."""

import numpy as np

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import make_locomotion_fly


def main() -> None:
    fly = make_locomotion_fly()
    world = FlatGroundWorld()
    world.add_fly(fly, (0, 0, 0.5), Rotation3D("quat", (1, 0, 0, 0)))
    simulation = Simulation(world)
    simulation.reset()
    simulation.set_leg_adhesion_states(fly.name, np.ones(6))
    simulation.step()
    positions = simulation.get_body_positions(fly.name)
    print(f"FlyGym smoke test passed: {positions.shape[0]} body segments, MuJoCo time={simulation.time:.4f}s")
    simulation.close()


if __name__ == "__main__":
    main()

