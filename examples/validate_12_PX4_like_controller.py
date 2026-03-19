#!/usr/bin/env python
"""
| File: validate_12_PX4_like_controller.py
| Description: Validation app for PX4-like controller with velocity CSV setpoint injection and tracking plots.
"""

import argparse
import math
import os
import sys
import csv
from pathlib import Path
from typing import Dict

import numpy as np

parser = argparse.ArgumentParser(description="PX4-like Torch controller example with configurable environment count.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of vehicle environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="XY spacing between spawned environments.")
parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode.")
parser.add_argument(
    "--control_mode",
    type=str,
    default="accel",
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
parser.add_argument("--vx", type=float, default=0.0, help="Constant velocity setpoint vx [m/s] in ENU.")
parser.add_argument("--vy", type=float, default=0.0, help="Constant velocity setpoint vy [m/s] in ENU.")
parser.add_argument("--vz", type=float, default=0.0, help="Constant velocity setpoint vz [m/s] in ENU.")
parser.add_argument("--yaw_rate", type=float, default=0.0, help="Constant yaw-rate setpoint [rad/s].")
parser.add_argument("--device", type=str, default="cpu", help="Torch device for controller math (for example: cpu, cuda:0).")
parser.add_argument("--data", action="store_true", help="Print simulation-fetched vx,vy,vz and ax,ay,az every physics update.")
parser.add_argument(
    "--velocity_setpoint_csv",
    type=str,
    default="",
    help="Velocity setpoint CSV path with columns (vx,vy,vz,yaw_rate[,phi]) or (t,vx,vy,vz,yaw_rate[,phi]).",
)
parser.add_argument(
    "--generate_test_velocity_csv",
    action="store_true",
    help="Generate the requested vz test profile CSV and use it as velocity setpoint source.",
)
parser.add_argument(
    "--generated_velocity_csv",
    type=str,
    default="trajectories/validate_vz_profile.csv",
    help="Output path for generated test profile CSV (relative to examples/ if not absolute).",
)
parser.add_argument(
    "--velocity_csv_dt",
    type=float,
    default=0.02,
    help="Sample dt used when CSV has no time column and for generated test profile.",
)
parser.add_argument(
    "--plot_output",
    type=str,
    default="results/validate_12_velocity_tracking.png",
    help="Output path for saved plot (relative to examples/ if not absolute).",
)
parser.add_argument(
    "--plot_env",
    type=int,
    default=0,
    help="Environment index to plot actual simulation values from.",
)
parser.add_argument(
    "--run_time",
    type=float,
    default=-1.0,
    help="Override run duration in seconds. Default: profile end if CSV profile is used, otherwise run until window close.",
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

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/utils")
from px4_like_controller import PX4LikeController

from scipy.spatial.transform import Rotation


def _resolve_examples_path(base_dir: str, maybe_relative: str) -> str:
    if maybe_relative == "":
        return ""
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.join(base_dir, maybe_relative)


def _read_velocity_csv(file_path: str, default_dt: float) -> Dict[str, np.ndarray]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(x) for x in row])
            except ValueError:
                # Skip headers/comments
                continue

    if len(rows) == 0:
        raise RuntimeError(f"No numeric setpoint rows found in: {file_path}")

    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] == 4:
        t = np.arange(data.shape[0], dtype=np.float64) * float(default_dt)
        vx, vy, vz, yaw_rate = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    elif data.shape[1] >= 5:
        # If first column is monotonic increasing, treat it as explicit time.
        first = data[:, 0]
        has_explicit_time = bool(np.all(np.diff(first) >= -1e-9))
        if has_explicit_time:
            t = first
            vx, vy, vz, yaw_rate = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        else:
            t = np.arange(data.shape[0], dtype=np.float64) * float(default_dt)
            vx, vy, vz, yaw_rate = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    else:
        raise RuntimeError("Velocity CSV must have at least 4 columns.")

    if t[0] > t[-1]:
        t = t[::-1]
        vx, vy, vz, yaw_rate = vx[::-1], vy[::-1], vz[::-1], yaw_rate[::-1]

    return {"t": t, "vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate}


def _generate_requested_profile_csv(file_path: str, dt: float):
    t = np.arange(0.0, 18.0 + dt, dt, dtype=np.float64)
    rng = np.random.default_rng(7)
    vz = np.zeros_like(t)

    # 0-4s: smooth ramp 0 -> 1.5 using smoothstep.
    idx0 = t < 4.0
    s = t[idx0] / 4.0
    smooth = 3.0 * s * s - 2.0 * s * s * s
    vz[idx0] = 1.5 * smooth

    # 4-8s: hold at 1.5.
    idx1 = (t >= 4.0) & (t < 8.0)
    vz[idx1] = 1.5

    # 8-14s: jump to -0.5 and hold.
    idx2 = (t >= 8.0) & (t < 14.0)
    vz[idx2] = -0.5

    # 14-18s: jump to 0 and hold.
    idx3 = t >= 14.0
    vz[idx3] = 0.0

    # Smooth random vx/vy/phi profiles (continuous, bounded).
    raw_vx = rng.normal(loc=0.0, scale=0.18, size=t.shape[0])
    raw_vy = rng.normal(loc=0.0, scale=0.14, size=t.shape[0])
    raw_phi = rng.normal(loc=0.0, scale=0.20, size=t.shape[0])

    win = max(5, int(round(0.4 / dt)))
    kernel = np.ones(win, dtype=np.float64) / float(win)

    vx = np.convolve(raw_vx, kernel, mode="same")
    vy = np.convolve(raw_vy, kernel, mode="same")
    phi = np.convolve(raw_phi, kernel, mode="same")

    # Keep values realistic and smooth for this validation test.
    vx = np.clip(vx, -0.35, 0.35)
    vy = np.clip(vy, -0.30, 0.30)
    phi = np.clip(phi, -0.40, 0.40)
    yaw_rate = np.gradient(phi, t, edge_order=1)
    yaw_rate = np.clip(yaw_rate, -0.8, 0.8)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for i in range(t.shape[0]):
            # Columns: t, vx, vy, vz, yaw_rate, phi
            writer.writerow([t[i], vx[i], vy[i], vz[i], yaw_rate[i], phi[i]])

    return {"t": t, "vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate, "phi": phi}


def _write_controller_trajectory_csv(profile: Dict[str, np.ndarray], file_path: str):
    t = profile["t"]
    vx = profile["vx"]
    vy = profile["vy"]
    vz = profile["vz"]
    yaw_rate = profile["yaw_rate"]

    ax = np.gradient(vx, t, edge_order=1)
    ay = np.gradient(vy, t, edge_order=1)
    az = np.gradient(vz, t, edge_order=1)
    jx = np.gradient(ax, t, edge_order=1)
    jy = np.gradient(ay, t, edge_order=1)
    jz = np.gradient(az, t, edge_order=1)

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    yaw = np.zeros_like(t)

    for i in range(1, t.shape[0]):
        dt = t[i] - t[i - 1]
        x[i] = x[i - 1] + vx[i - 1] * dt
        y[i] = y[i - 1] + vy[i - 1] * dt
        z[i] = z[i - 1] + vz[i - 1] * dt
        yaw[i] = yaw[i - 1] + yaw_rate[i - 1] * dt

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for i in range(t.shape[0]):
            writer.writerow(
                [
                    t[i],
                    x[i],
                    y[i],
                    z[i],
                    vx[i],
                    vy[i],
                    vz[i],
                    ax[i],
                    ay[i],
                    az[i],
                    jx[i],
                    jy[i],
                    jz[i],
                    yaw[i],
                    yaw_rate[i],
                ]
            )


def _sample_profile(profile: Dict[str, np.ndarray], t_now: float):
    t = profile["t"]
    vx = float(np.interp(t_now, t, profile["vx"]))
    vy = float(np.interp(t_now, t, profile["vy"]))
    vz = float(np.interp(t_now, t, profile["vz"]))
    yaw_rate = float(np.interp(t_now, t, profile["yaw_rate"]))
    return vx, vy, vz, yaw_rate


class PegasusApp:
    """Standalone app for running multiple multirotors with a PX4-like Torch controller."""

    def __init__(self, num_envs: int = 1, env_spacing: float = 2.5):
        self.num_envs = num_envs
        self.env_spacing = env_spacing

        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        self._setup_lighting()

        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())
        self.results_dir = self.curr_dir + "/results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.physics_dt = float(self.pg._world_settings["physics_dt"])

        self.profile = None
        self.controller_trajectory_file = ""
        self.stop_time = None
        self.vehicles = []
        self.log_t = []
        self.log_v = []
        self.log_a = []
        self.log_v_sp = []
        self.log_yaw_rate_sp = []

        csv_path = _resolve_examples_path(self.curr_dir, args_cli.velocity_setpoint_csv)
        if args_cli.generate_test_velocity_csv:
            csv_path = _resolve_examples_path(self.curr_dir, args_cli.generated_velocity_csv)
            self.profile = _generate_requested_profile_csv(csv_path, args_cli.velocity_csv_dt)
            carb.log_warn(f"Generated velocity profile CSV at: {csv_path}")
        elif csv_path != "":
            self.profile = _read_velocity_csv(csv_path, args_cli.velocity_csv_dt)

        if self.profile is not None:
            self.controller_trajectory_file = os.path.join(self.results_dir, "validate_velocity_profile_controller.csv")
            _write_controller_trajectory_csv(self.profile, self.controller_trajectory_file)
            self.trajectory_file = self.controller_trajectory_file
            self.stop_time = float(self.profile["t"][-1])
        elif args_cli.use_trajectory:
            if os.path.isabs(args_cli.trajectory_file):
                self.trajectory_file = args_cli.trajectory_file
            else:
                self.trajectory_file = os.path.join(self.curr_dir, args_cli.trajectory_file)
        else:
            self.trajectory_file = None

        if args_cli.run_time > 0.0:
            self.stop_time = args_cli.run_time

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

        vehicle = Multirotor(
            f"/World/quadrotor{env_id}",
            ROBOTS["Iris"],
            env_id,
            init_pos,
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )
        self.vehicles.append(vehicle)

    def _results_file(self, env_id: int):
        if self.num_envs == 1:
            return self.results_dir + "/px4_like_statistics.npz"
        return self.results_dir + f"/px4_like_statistics_env_{env_id:03d}.npz"

    def _log_validation_sample(self, t_now: float):
        if len(self.vehicles) == 0:
            return

        env_index = max(0, min(args_cli.plot_env, len(self.vehicles) - 1))
        state = self.vehicles[env_index].state
        self.log_t.append(t_now)
        self.log_v.append(np.array(state.linear_velocity, dtype=np.float64))
        self.log_a.append(np.array(state.linear_acceleration, dtype=np.float64))

        if self.profile is not None:
            vx_sp, vy_sp, vz_sp, yaw_rate_sp = _sample_profile(self.profile, t_now)
        else:
            vx_sp, vy_sp, vz_sp, yaw_rate_sp = args_cli.vx, args_cli.vy, args_cli.vz, args_cli.yaw_rate

        self.log_v_sp.append(np.array([vx_sp, vy_sp, vz_sp], dtype=np.float64))
        self.log_yaw_rate_sp.append(float(yaw_rate_sp))

    def _save_validation_plot(self):
        if len(self.log_t) == 0:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            carb.log_warn(f"Could not import matplotlib, skipping plot: {exc}")
            return

        t = np.asarray(self.log_t)
        v = np.vstack(self.log_v)
        a = np.vstack(self.log_a)
        v_sp = np.vstack(self.log_v_sp)

        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

        axes[0].plot(t, v[:, 0], label="vx actual")
        axes[0].plot(t, v_sp[:, 0], "--", label="vx setpoint")
        axes[0].set_ylabel("vx [m/s]")
        axes[0].grid(True)
        axes[0].legend(loc="best")

        axes[1].plot(t, v[:, 1], label="vy actual")
        axes[1].plot(t, v_sp[:, 1], "--", label="vy setpoint")
        axes[1].set_ylabel("vy [m/s]")
        axes[1].grid(True)
        axes[1].legend(loc="best")

        axes[2].plot(t, v[:, 2], label="vz actual")
        axes[2].plot(t, v_sp[:, 2], "--", label="vz setpoint")
        axes[2].set_ylabel("vz [m/s]")
        axes[2].grid(True)
        axes[2].legend(loc="best")

        axes[3].plot(t, a[:, 0], label="ax actual")
        axes[3].plot(t, a[:, 1], label="ay actual")
        axes[3].plot(t, a[:, 2], label="az actual")
        axes[3].set_ylabel("a [m/s^2]")
        axes[3].set_xlabel("time [s]")
        axes[3].grid(True)
        axes[3].legend(loc="best")

        fig.suptitle(
            f"PX4-like validation (env {max(0, min(args_cli.plot_env, len(self.vehicles)-1))})\n"
            f"physics_dt={self.physics_dt:.6f}s ({1.0/self.physics_dt:.1f} Hz)"
        )
        fig.tight_layout()

        output_path = _resolve_examples_path(self.curr_dir, args_cli.plot_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        carb.log_warn(f"Saved validation plot: {output_path}")

    def run(self):
        self.timeline.play()
        sim_time = 0.0
        while simulation_app.is_running():
            self.world.step(render=not args_cli.headless)
            sim_time += self.physics_dt
            self._log_validation_sample(sim_time)

            if self.stop_time is not None and sim_time >= self.stop_time:
                break

        carb.log_warn("PegasusApp Simulation App is closing.")
        self._save_validation_plot()
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    pg_app.run()


if __name__ == "__main__":
    main()
