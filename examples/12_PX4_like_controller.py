#!/usr/bin/env python
"""
| File: 12_PX4_like_controller.py
| Description: Example that runs a PX4-like multicopter cascade in Torch with selectable accel or velocity input mode.
"""

import argparse
import math
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="PX4-like Torch controller example with configurable environment count.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of vehicle environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="XY spacing between spawned environments.")
parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode.")
parser.add_argument(
    "--control_mode",
    type=str,
    default="velocity",
    choices=("accel", "velocity"),
    help="Select input mode: accel=(ax,ay,az,yaw_rate) or velocity=(vx,vy,vz,yaw_rate).",
)

parser.add_argument(
    "--use_trajectory",
    action="store_true",
    help="Use trajectory CSV setpoints. velocity mode uses columns [4:7]; accel mode uses [7:10]; yaw_rate uses [14].",
)
parser.add_argument(
    "--trajectory_file",
    type=str,
    default="trajectories/pitch_relay_90_deg_2.csv",
    help="Trajectory CSV path. If relative, it is resolved from the examples folder.",
)
parser.add_argument("--ax", type=float, default=0.0, help="Constant acceleration setpoint ax [m/s^2] in ENU.")
parser.add_argument("--ay", type=float, default=0.0, help="Constant acceleration setpoint ay [m/s^2] in ENU.")
parser.add_argument("--az", type=float, default=0.0, help="Constant acceleration setpoint az [m/s^2] in ENU.")
parser.add_argument("--vx", type=float, default=0.2, help="Constant velocity setpoint vx [m/s] in ENU.")
parser.add_argument("--vy", type=float, default=0.0, help="Constant velocity setpoint vy [m/s] in ENU.")
parser.add_argument("--vz", type=float, default=0.2, help="Constant velocity setpoint vz [m/s] in ENU.")
parser.add_argument("--yaw_rate", type=float, default=0.0, help="Constant yaw-rate setpoint [rad/s].")
parser.add_argument("--device", type=str, default="cpu", help="Torch device for controller math (for example: cpu, cuda:0).")
parser.add_argument("--data", action="store_true", help="Print simulation-fetched vx,vy,vz and ax,ay,az every physics update.")
parser.add_argument(
    "--platform_texture",
    type=str,
    default="/home/rycker/projects/ros2_ws/src/stewart_platform_learning/src/stewart_platform/aruco_visual_marker_fractal/materials/textures/aruco_mark_fractal.png",
    help="PNG file path to project on the top of /World/platform. If relative, it is resolved from the examples folder.",
)

args_cli, _ = parser.parse_known_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be greater than or equal to 1.")

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args_cli.headless})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils
from pxr import Sdf, UsdGeom, UsdShade

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/utils")
from px4_like_controller import PX4LikeController

from scipy.spatial.transform import Rotation


class PegasusApp:
    """Standalone app for running multiple multirotors with a PX4-like Torch controller."""

    def __init__(self, num_envs: int = 1, env_spacing: float = 2.5):
        self.num_envs = num_envs
        self.env_spacing = env_spacing

        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())
        if os.path.isabs(args_cli.platform_texture):
            self.platform_texture = args_cli.platform_texture
        else:
            self.platform_texture = str((Path(self.curr_dir) / args_cli.platform_texture).resolve())

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        self._setup_lighting()
        self._setup_platform()

        self.results_dir = self.curr_dir + "/results"
        os.makedirs(self.results_dir, exist_ok=True)

        if args_cli.use_trajectory:
            if os.path.isabs(args_cli.trajectory_file):
                self.trajectory_file = args_cli.trajectory_file
            else:
                self.trajectory_file = os.path.join(self.curr_dir, args_cli.trajectory_file)
        else:
            self.trajectory_file = None

        for env_id in range(self.num_envs):
            self._spawn_vehicle(env_id)

        self.world.reset()

    def _setup_lighting(self):
        prim_utils.create_prim(
            "/World/Light/DomeLight",
            "DomeLight",
            attributes={
                "inputs:intensity": 4500.0,
                "inputs:color": (1.0, 1.0, 1.0),
                "inputs:texture:file": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr",
            },
        )
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

    def _setup_platform(self):
        # 1x1x0.2 static box centered at z=0.1 so it sits on the floor, named "platform".
        platform_path = "/World/platform"
        prim_utils.create_prim(
            platform_path,
            "Cube",
            position=(0.0, 0.0, 0.1),
            scale=(1.0, 1.0, 0.2),
            attributes={
                "size": 1.0,
                "primvars:displayColor": [(0.28, 0.28, 0.28)],
            },
        )

        if not os.path.isfile(self.platform_texture):
            carb.log_warn(
                "Platform texture file not found at: "
                + self.platform_texture
                + ". The platform will be created without a top image."
            )
            return

        stage = self.world.stage

        # Add a thin quad just above the top face so only the top shows the PNG.
        decal_mesh_path = Sdf.Path("/World/platform/top_decal")
        decal_mesh = UsdGeom.Mesh.Define(stage, decal_mesh_path)
        # Local Z in the platform frame: top face is at +0.5 for a unit cube.
        decal_z = 0.5005
        decal_mesh.CreatePointsAttr(
            [
                (-0.5, -0.5, decal_z),
                (0.5, -0.5, decal_z),
                (0.5, 0.5, decal_z),
                (-0.5, 0.5, decal_z),
            ]
        )
        decal_mesh.CreateFaceVertexCountsAttr([4])
        decal_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        decal_mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 4)
        decal_mesh.SetNormalsInterpolation("vertex")
        decal_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        primvars_api = UsdGeom.PrimvarsAPI(decal_mesh)
        st_primvar = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st_primvar.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        material_path = Sdf.Path("/World/Looks/platform_top_material")
        material = UsdShade.Material.Define(stage, material_path)

        pbr_shader = UsdShade.Shader.Define(stage, material_path.AppendPath("PreviewSurface"))
        pbr_shader.CreateIdAttr("UsdPreviewSurface")
        pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        uv_reader = UsdShade.Shader.Define(stage, material_path.AppendPath("PrimvarReader_st"))
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        tex_shader = UsdShade.Shader.Define(stage, material_path.AppendPath("TopTexture"))
        tex_shader.CreateIdAttr("UsdUVTexture")
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(self.platform_texture))
        tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex_shader.ConnectableAPI(), "rgb"
        )
        material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(decal_mesh.GetPrim()).Bind(material)

    def _spawn_vehicle(self, env_id: int):
        side = math.ceil(math.sqrt(self.num_envs))
        row = env_id // side
        col = env_id % side

        init_pos = [
            2.3 + (col * self.env_spacing),
            -1.5 + (row * self.env_spacing),
            0.07,
        ]

        config_multirotor = MultirotorConfig()
        config_multirotor.sensors = []
        config_multirotor.backends = [
            PX4LikeController(
                trajectory_file=self.trajectory_file,
                results_file=self._results_file(env_id),
                input_mode=args_cli.control_mode,
                velocity_command=(args_cli.vx, args_cli.vy, args_cli.vz),
                accel_command=(args_cli.ax, args_cli.ay, args_cli.az),
                yaw_rate_command=args_cli.yaw_rate,
                device=args_cli.device,
                data=args_cli.data,
            )
        ]

        Multirotor(
            f"/World/quadrotor{env_id}",
            ROBOTS["Iris"],
            env_id,
            init_pos,
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

    def _results_file(self, env_id: int):
        if self.num_envs == 1:
            return self.results_dir + "/px4_like_statistics.npz"
        return self.results_dir + f"/px4_like_statistics_env_{env_id:03d}.npz"

    def run(self):
        self.timeline.play()
        while simulation_app.is_running():
            self.world.step(render=not args_cli.headless)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    pg_app.run()


if __name__ == "__main__":
    main()
