#!/usr/bin/env python
"""
| File: 13_ardu_ros.py
| Description: Multi-vehicle ArduPilot example with a ROS 2 velocity-command bridge.
"""

import argparse
import atexit
import os
import signal
import subprocess
import time
import numpy as np

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(
    description="Run multiple ArduPilot-backed vehicles and command them through ROS 2 velocity topics."
)
parser.add_argument("--num_drones", type=int, default=1, help="Number of drones to spawn.")
parser.add_argument("--gap_x_axis", type=float, default=1.0, help="Spacing between vehicles along x [m].")
parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
parser.add_argument("--namespace", type=str, default="drone", help="ROS 2 topic namespace prefix.")
parser.add_argument(
    "--bridge_baseport",
    type=int,
    default=14650,
    help="Base UDP port used by the ROS 2 MAVLink bridge. Each vehicle uses baseport + 10 * vehicle_id.",
)
parser.add_argument(
    "--auto_takeoff_alt",
    type=float,
    default=2.0,
    help="Automatic takeoff altitude [m]. Set to 0 to disable automatic takeoff.",
)
parser.add_argument(
    "--cmd_timeout",
    type=float,
    default=0.5,
    help="If no ROS 2 cmd_vel arrives for this many seconds, send zero velocity.",
)
parser.add_argument(
    "--arm_delay",
    type=float,
    default=3.0,
    help="Minimum delay after the first ArduPilot heartbeat before attempting GUIDED arm/takeoff.",
)
parser.add_argument(
    "--strict_prearm",
    action="store_true",
    help="Keep ArduPilot's default EKF/pre-arm startup instead of using the fast SITL AHRS mode.",
)
parser.add_argument(
    "--cleanup",
    action="store_true",
    help="Kill ArduPilot SITL, sim_vehicle.py, and MAVProxy processes and exit.",
)
parser.add_argument(
    "--show_ardupilot_ui",
    action="store_true",
    help="Launch each ArduPilot instance in its own terminal with MAVProxy console and map.",
)

args_cli, _ = parser.parse_known_args()
if args_cli.num_drones < 1:
    parser.error("--num_drones must be greater than or equal to 1.")

FAST_START_PARAM_TEXT = "\n".join(
    [
        "AHRS_EKF_TYPE 10",
        "EK2_ENABLE 0",
        "EK3_ENABLE 0",
        "",
    ]
)


def _matching_ardupilot_process(cmdline: str) -> bool:
    cmdline_lower = cmdline.lower()
    return (
        "sim_vehicle.py" in cmdline_lower
        or "mavproxy.py" in cmdline_lower
        or "arducopter" in cmdline_lower
    )


def _get_process_table():
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.SubprocessError:
        return {}

    table = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        table[pid] = {"ppid": ppid, "cmdline": parts[2]}
    return table


def _descendants(process_table, root_pid: int):
    children = {}
    for pid, info in process_table.items():
        children.setdefault(info["ppid"], []).append(pid)

    ordered = []
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        stack.extend(children.get(pid, []))
    return list(reversed(ordered))


def _terminate_process_tree(root_pid: int, timeout: float = 5.0, process_table=None) -> int:
    if process_table is None:
        process_table = _get_process_table()

    if root_pid not in process_table:
        return 0

    targets = _descendants(process_table, root_pid)
    if not targets:
        return 0

    try:
        os.killpg(os.getpgid(root_pid), signal.SIGINT)
    except OSError:
        for pid in targets:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [pid for pid in targets if os.path.exists(f"/proc/{pid}")]
        if not alive:
            return len(targets)
        time.sleep(0.1)

    try:
        os.killpg(os.getpgid(root_pid), signal.SIGKILL)
    except OSError:
        for pid in targets:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    time.sleep(0.2)
    return len(targets)


def cleanup_ardupilot_processes() -> int:
    process_table = _get_process_table()
    root_pids = [
        pid for pid, info in process_table.items() if _matching_ardupilot_process(info["cmdline"])
    ]

    cleaned = 0
    for pid in sorted(set(root_pids)):
        cleaned += _terminate_process_tree(pid, timeout=2.0, process_table=process_table)

    return cleaned


if args_cli.cleanup:
    cleaned = cleanup_ardupilot_processes()
    print(f"[13_ardu_ros] cleanup complete, terminated {cleaned} process entries.")
    raise SystemExit(0)

simulation_app = SimulationApp({"headless": args_cli.headless})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

from geometry_msgs.msg import Twist
from pymavlink import mavutil

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, WORLD_SETTINGS
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend,
    ArduPilotMavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.backends.tools.ardupilot_launch_tool import ArduPilotLaunchTool
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation
import rclpy


class MultiOutArduPilotLaunchTool(ArduPilotLaunchTool):
    """Launch ArduPilot SITL with two MAVProxy outputs: one for Pegasus, one for the ROS bridge."""

    def __init__(
        self,
        ardupilot_dir,
        vehicle_id: int,
        ardupilot_model: str,
        out_ports,
        fast_start: bool = True,
        show_ui: bool = False,
    ):
        super().__init__(ardupilot_dir, vehicle_id, ardupilot_model)
        self.out_ports = list(out_ports)
        self.fast_start = fast_start
        self.show_ui = show_ui
        self._fast_start_param_file = None

    def _ensure_fast_start_param_file(self):
        if self._fast_start_param_file is not None:
            return self._fast_start_param_file

        param_path = os.path.join(self.root_fs.name, f"pegasus_fast_start_{self.vehicle_id}.parm")
        with open(param_path, "w", encoding="ascii") as param_file:
            param_file.write(FAST_START_PARAM_TEXT)
        self._fast_start_param_file = param_path
        return self._fast_start_param_file

    def launch_ardupilot(self):
        command = [
            "python3",
            f"{self.ardupilot_dir}/Tools/autotest/sim_vehicle.py",
            "-v",
            "ArduCopter",
            "-f",
            self._get_vehicle_frame(),
            "--model",
            self.model,
            "-I",
            str(self.vehicle_id),
            "--sysid",
            str(self.vehicle_id + 1),
        ]

        if self._sitl_already_exists():
            command.append("--no-rebuild")

        if self.fast_start:
            command.extend(["--add-param-file", self._ensure_fast_start_param_file()])

        if self.show_ui:
            command.extend(["--console", "--map"])

        for port in self.out_ports:
            command.extend(["--out", f"udp:127.0.0.1:{port}"])

        if self.show_ui:
            command_str = " ".join(command)
            self.ardupilot_process = subprocess.Popen(
                ["gnome-terminal", "--", "bash", "-lc", command_str],
                cwd=self.root_fs.name,
                shell=False,
                env=self.environment,
                preexec_fn=os.setsid,
            )
            return

        self.ardupilot_process = subprocess.Popen(
            command,
            cwd=self.root_fs.name,
            shell=False,
            env=self.environment,
            preexec_fn=os.setsid,
        )

    def kill_ardupilot(self):
        if self.ardupilot_process is None:
            return

        _terminate_process_tree(self.ardupilot_process.pid, timeout=5.0)
        self.ardupilot_process = None


class ArduPilotRos2VelocityBridge(Backend):
    """ROS 2 backend that forwards geometry_msgs/Twist commands to ArduPilot GUIDED velocity control."""

    def __init__(
        self,
        vehicle_id: int,
        namespace: str = "drone",
        bridge_baseport: int = 14650,
        auto_takeoff_alt: float = 2.0,
        cmd_timeout: float = 0.5,
        send_rate_hz: float = 20.0,
        num_rotors: int = 4,
    ):
        super().__init__(config=None)

        self._vehicle_id = vehicle_id
        self._namespace = namespace
        self._bridge_port = bridge_baseport + vehicle_id * 10
        self._auto_takeoff_alt = max(float(auto_takeoff_alt), 0.0)
        self._cmd_timeout = max(float(cmd_timeout), 0.0)
        self._send_period = 1.0 / max(float(send_rate_hz), 1.0)
        self._input_ref = [0.0 for _ in range(num_rotors)]
        self._arm_delay = max(float(args_cli.arm_delay), 0.0)
        self._require_position_ready = bool(args_cli.strict_prearm)

        self._connection = None
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._current_altitude = 0.0
        self._body_to_local_ned = Rotation.identity()
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_wait_log_time = 0.0

        self._latest_cmd = Twist()
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(f"ardupilot_ros2_bridge_{vehicle_id}")
        topic = f"{self._namespace}{self._vehicle_id}/cmd_vel"
        self._cmd_sub = self.node.create_subscription(Twist, topic, self._cmd_vel_callback, 10)

        carb.log_warn(
            f"[ArduPilotRos2VelocityBridge] vehicle_id={vehicle_id} listening on ROS 2 topic "
            f"'{topic}' and MAVLink bridge port {self._bridge_port}"
        )

    def _cmd_vel_callback(self, msg: Twist):
        self._latest_cmd = msg
        self._last_cmd_time = time.monotonic()

    def _drain_mavlink(self):
        if self._connection is None:
            return

        while True:
            msg = self._connection.recv_match(blocking=False)
            if msg is None:
                break

            if msg.get_type() == "HEARTBEAT" and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                self._connected = True
                self._target_system = msg.get_srcSystem()
                self._target_component = msg.get_srcComponent()
                if self._first_heartbeat_time is None:
                    self._first_heartbeat_time = time.monotonic()
            elif msg.get_type() == "GPS_RAW_INT":
                self._gps_fix_ready = getattr(msg, "fix_type", 0) >= 3
            elif msg.get_type() in ("GLOBAL_POSITION_INT", "LOCAL_POSITION_NED"):
                self._position_estimate_ready = True

    def _send_command_long(self, command, params):
        if self._connection is None or self._target_system is None or self._target_component is None:
            return

        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            command,
            0,
            *params,
        )

    def _update_takeoff_state(self, now: float):
        if not self._connected:
            return

        if self._takeoff_state == "ready":
            return

        if self._first_heartbeat_time is None:
            return

        wait_reasons = []
        if (now - self._first_heartbeat_time) < self._arm_delay:
            wait_reasons.append(f"arm_delay {self._arm_delay:.1f}s")
        if not self._gps_fix_ready:
            wait_reasons.append("gps_fix")
        if self._require_position_ready and not self._position_estimate_ready:
            wait_reasons.append("position_estimate")

        if wait_reasons:
            if now - self._last_wait_log_time >= 2.0:
                carb.log_warn(
                    f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: waiting for "
                    + ", ".join(wait_reasons)
                )
                self._last_wait_log_time = now
            return

        # Step through GUIDED -> ARM -> TAKEOFF with light retries.
        if self._takeoff_state == "pending":
            self._connection.set_mode_apm("GUIDED")
            self._takeoff_state = "arming"
            self._last_action_time = now
            carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: requested GUIDED mode")
            return

        if self._takeoff_state == "arming":
            if self._connection.motors_armed():
                self._send_command_long(
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._auto_takeoff_alt],
                )
                self._takeoff_state = "taking_off"
                self._last_action_time = now
                carb.log_warn(
                    f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: armed, taking off to {self._auto_takeoff_alt:.2f} m"
                )
                return

            if now - self._last_action_time >= 1.0:
                self._connection.set_mode_apm("GUIDED")
                self._connection.arducopter_arm()
                self._last_action_time = now
                carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: retrying arm")
            return

        if self._takeoff_state == "taking_off":
            if self._current_altitude >= 0.8 * self._auto_takeoff_alt:
                self._takeoff_state = "ready"
                carb.log_warn(f"[ArduPilotRos2VelocityBridge] drone{self._vehicle_id}: ready for velocity commands")
                return

            if now - self._last_action_time >= 2.0:
                self._send_command_long(
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._auto_takeoff_alt],
                )
                self._last_action_time = now

    def _send_velocity_command(self, now: float):
        if not self._connected or self._takeoff_state != "ready":
            return

        if now - self._last_send_time < self._send_period:
            return

        cmd_age = now - self._last_cmd_time
        if self._last_cmd_time == 0.0 or cmd_age > self._cmd_timeout:
            vx = 0.0
            vy = 0.0
            vz = 0.0
            yaw_rate = 0.0
        else:
            # ROS Twist is interpreted in ROS body FLU.
            # Convert to local NED to match ArduPilot's well-tested guided velocity path.
            vel_body_frd = np.array(
                [
                    float(self._latest_cmd.linear.x),
                    -float(self._latest_cmd.linear.y),
                    -float(self._latest_cmd.linear.z),
                ]
            )
            vel_local_ned = self._body_to_local_ned.apply(vel_body_frd)
            vx = float(vel_local_ned[0])
            vy = float(vel_local_ned[1])
            vz = float(vel_local_ned[2])
            yaw_rate = -float(self._latest_cmd.angular.z)

        # Match ArduPilot's known-good guided velocity path: position ignored,
        # velocity provided, acceleration explicitly zeroed, yaw ignored,
        # yaw-rate optionally commanded.
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        self._connection.mav.set_position_target_local_ned_send(
            int(now * 1000.0) & 0xFFFFFFFF,
            self._target_system,
            self._target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0.0,
            0.0,
            0.0,
            vx,
            vy,
            vz,
            0.0,
            0.0,
            0.0,
            0.0,
            yaw_rate,
        )
        self._last_send_time = now

    def update_sensor(self, sensor_type: str, data):
        return

    def update_graphical_sensor(self, sensor_type: str, data):
        return

    def update_state(self, state):
        self._current_altitude = float(state.position[2])
        self._body_to_local_ned = Rotation.from_quat(state.get_attitude_ned_frd())

    def input_reference(self):
        return self._input_ref

    def update(self, dt: float):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._drain_mavlink()

        now = time.monotonic()
        self._update_takeoff_state(now)
        self._send_velocity_command(now)

    def start(self):
        self._connection = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{self._bridge_port}",
            source_system=200 + self._vehicle_id,
        )
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._last_wait_log_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"

    def stop(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def reset(self):
        self._latest_cmd = Twist()
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._first_heartbeat_time = None
        self._gps_fix_ready = False
        self._position_estimate_ready = False
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._last_action_time = 0.0
        self._last_wait_log_time = 0.0
        self._takeoff_state = "pending" if self._auto_takeoff_alt > 0.0 else "ready"


class PegasusApp:
    """Standalone app for running multiple ArduPilot vehicles with ROS 2 velocity command bridges."""

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world_settings = dict(WORLD_SETTINGS["ardupilot"])
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        self.stop_sim = False
        self._shutdown_complete = False
        self._signal_count = 0
        self.ardupilot_tools = []
        self.ardupilot_started = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        atexit.register(self._shutdown)

        for vehicle_id in range(args_cli.num_drones):
            self.vehicle_factory(vehicle_id, gap_x_axis=args_cli.gap_x_axis)

        self.world.reset()

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float):
        config_multirotor = MultirotorConfig()

        backend_config = ArduPilotMavlinkBackendConfig(
            {
                "vehicle_id": vehicle_id,
                "ardupilot_autolaunch": False,
                "ardupilot_dir": self.pg.ardupilot_path,
                "ardupilot_vehicle_model": "gazebo-iris",
            }
        )

        config_multirotor.backends = [
            ArduPilotMavlinkBackend(config=backend_config),
            ROS2Backend(
                vehicle_id=vehicle_id,
                config={
                    "namespace": args_cli.namespace,
                    "pub_sensors": True,
                    "pub_graphical_sensors": True,
                    "pub_state": True,
                    "sub_control": False,
                    "pub_tf": True,
                },
            ),
            ArduPilotRos2VelocityBridge(
                vehicle_id=vehicle_id,
                namespace=args_cli.namespace,
                bridge_baseport=args_cli.bridge_baseport,
                auto_takeoff_alt=args_cli.auto_takeoff_alt,
                cmd_timeout=args_cli.cmd_timeout,
                num_rotors=config_multirotor.thrust_curve._num_rotors,
            ),
        ]

        self.ardupilot_tools.append(
            MultiOutArduPilotLaunchTool(
                ardupilot_dir=self.pg.ardupilot_path,
                vehicle_id=vehicle_id,
                ardupilot_model="gazebo-iris",
                out_ports=[
                    14550 + vehicle_id * 10,
                    args_cli.bridge_baseport + vehicle_id * 10,
                ],
                fast_start=not args_cli.strict_prearm,
                show_ui=args_cli.show_ardupilot_ui,
            )
        )

        Multirotor(
            f"/World/drone{vehicle_id}",
            ROBOTS["Iris"],
            vehicle_id,
            [gap_x_axis * vehicle_id, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

    def _launch_ardupilot(self):
        if self.ardupilot_started:
            return

        for launch_tool in self.ardupilot_tools:
            launch_tool.launch_ardupilot()

        self.ardupilot_started = True
        carb.log_warn("[13_ardu_ros] ArduPilot SITL instances launched")

    def _kill_ardupilot(self):
        if not self.ardupilot_started:
            return

        for launch_tool in self.ardupilot_tools:
            try:
                launch_tool.kill_ardupilot()
            except Exception as exc:
                carb.log_warn(f"[13_ardu_ros] Failed to kill ArduPilot instance cleanly: {exc}")

        self.ardupilot_started = False

    def _handle_signal(self, signum, _frame):
        self._signal_count += 1
        if self._signal_count == 1:
            carb.log_warn(f"[13_ardu_ros] received signal {signum}, stopping simulation")
            self.stop_sim = True
            return

        carb.log_warn(f"[13_ardu_ros] received second signal {signum}, forcing cleanup")
        self._shutdown()
        raise SystemExit(128 + signum)

    def _shutdown(self):
        if self._shutdown_complete:
            return

        self._shutdown_complete = True
        self.stop_sim = True

        try:
            self._kill_ardupilot()
        except Exception as exc:
            carb.log_warn(f"[13_ardu_ros] Failed while stopping ArduPilot: {exc}")

        try:
            self.timeline.stop()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass

    def run(self):
        self.timeline.play()
        self._launch_ardupilot()

        try:
            while simulation_app.is_running() and not self.stop_sim:
                self.world.step(render=not args_cli.headless)
        except KeyboardInterrupt:
            self.stop_sim = True
        finally:
            carb.log_warn("PegasusApp Simulation App is closing.")
            self._shutdown()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
