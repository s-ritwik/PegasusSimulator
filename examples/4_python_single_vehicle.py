#!/usr/bin/env python
"""
| File: 4_python_single_vehicle.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller 
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS.
"""

import argparse
import math
import os
import sys

# add argparse arguments
parser = argparse.ArgumentParser(description="Python control backend example with configurable environment count.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of vehicle environments to spawn.")
args_cli, _ = parser.parse_known_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be greater than or equal to 1.")

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Import the custom python control backend
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/utils')
from nonlinear_controller import NonlinearController

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

# Use pathlib for parsing the desired trajectory from a CSV file
from pathlib import Path


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self, num_envs: int = 1, env_spacing: float = 2.5):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """
        self.num_envs = num_envs
        self.env_spacing = env_spacing

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        self._setup_lighting()

        # Get the current directory used to read trajectories and save results
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())
        self.trajectory_file = self.curr_dir + "/trajectories/pitch_relay_90_deg_2.csv"
        self.results_dir = self.curr_dir + "/results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Spawn multiple vehicles using a regular XY grid layout
        for env_id in range(self.num_envs):
            self._spawn_vehicle(env_id)

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

    def _setup_lighting(self):
        """Add a brighter and more cinematic lighting setup for the scene."""

        # Global image-based lighting for soft ambient illumination.
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            attributes={
                "inputs:intensity": 4500.0,
                "inputs:color": (1.0, 1.0, 1.0),
                "inputs:texture:file": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr",
            },
        )

        # Key light: warm and bright to highlight vehicles.
        prim_utils.create_prim(
            "/World/Light/KeySphere",
            "SphereLight",
            position=(7.0, -2.0, 6.5),
            attributes={
                "inputs:intensity": 25000.0,
                "inputs:radius": 1.25,
                "inputs:color": (1.0, 0.96, 0.92),
            },
        )

        # Fill light: softer cool light to reduce hard shadows.
        prim_utils.create_prim(
            "/World/Light/FillSphere",
            "SphereLight",
            position=(-1.5, 4.5, 4.2),
            attributes={
                "inputs:intensity": 12000.0,
                "inputs:radius": 1.0,
                "inputs:color": (0.84, 0.90, 1.0),
            },
        )

    def _spawn_vehicle(self, env_id: int):
        """Spawn a vehicle/controller pair for a given environment index."""
        side = math.ceil(math.sqrt(self.num_envs))
        row = env_id // side
        col = env_id % side

        init_pos = [
            2.3 + (col * self.env_spacing),
            -1.5 + (row * self.env_spacing),
            0.07,
        ]

        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        # This example controller uses state feedback directly; disabling sensors avoids extra per-step overhead.
        config_multirotor.sensors = []
        config_multirotor.backends = [NonlinearController(
            trajectory_file=self.trajectory_file,
            results_file=self._results_file(env_id),
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0]
        )]

        Multirotor(
            f"/World/quadrotor{env_id}",
            ROBOTS['Iris'],
            env_id,
            init_pos,
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

    def _results_file(self, env_id: int):
        if self.num_envs == 1:
            return self.results_dir + "/single_statistics.npz"
        return self.results_dir + f"/single_statistics_env_{env_id:03d}.npz"

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running():

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp(num_envs=args_cli.num_envs)

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
