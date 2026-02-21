#!/usr/bin/env python
"""
| File: px4_like_controller.py
| Description: Pegasus backend implementing a PX4-like multicopter control cascade in Torch.
"""

import csv
from typing import Optional

import carb
import numpy as np
import torch

from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.state import State

from px4_like_pipeline import HilActuatorMapper, PX4LikeMulticopterCascade, yaw_from_quaternion


class MulticopterCommandSource:
    """Provides velocity/acceleration/yaw-rate setpoints, either from constants or a trajectory CSV."""

    def __init__(
        self,
        trajectory_file: Optional[str],
        velocity_command,
        accel_command,
        yaw_rate_command: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype

        self._constant_velocity = torch.as_tensor(velocity_command, device=device, dtype=dtype)
        self._constant_accel = torch.as_tensor(accel_command, device=device, dtype=dtype)
        self._constant_yaw_rate = torch.as_tensor(yaw_rate_command, device=device, dtype=dtype)

        self._trajectory = self._load_trajectory(trajectory_file) if trajectory_file is not None else None
        self._index = 0

    def reset(self):
        self._index = 0

    def _load_trajectory(self, file_path: str) -> torch.Tensor:
        rows = []
        with open(file_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            for row in reader:
                if len(row) < 15:
                    continue
                try:
                    rows.append([float(entry) for entry in row])
                except ValueError:
                    continue

        if len(rows) == 0:
            raise RuntimeError(f"Trajectory file has no valid numeric rows: {file_path}")

        data = torch.tensor(rows, device=self.device, dtype=self.dtype)

        # Existing repo trajectories are typically in descending time.
        if data[0, 0] > data[-1, 0]:
            data = torch.flip(data, dims=(0,))

        # [time, vx, vy, vz, ax, ay, az, yaw_rate]
        return torch.stack((data[:, 0], data[:, 4], data[:, 5], data[:, 6], data[:, 7], data[:, 8], data[:, 9], data[:, 14]), dim=1)

    def _sample_trajectory(self, t_now: float):
        max_index = int(self._trajectory.shape[0] - 1)
        while self._index < max_index and t_now >= float(self._trajectory[self._index + 1, 0].item()):
            self._index += 1

        if self._index >= max_index:
            row = self._trajectory[max_index]
            return row

        row0 = self._trajectory[self._index]
        row1 = self._trajectory[self._index + 1]
        t0 = float(row0[0].item())
        t1 = float(row1[0].item())
        if t1 <= t0:
            return row0

        alpha = (t_now - t0) / (t1 - t0)
        alpha = min(max(alpha, 0.0), 1.0)
        alpha_t = torch.as_tensor(alpha, device=self.device, dtype=self.dtype)
        return row0 * (1.0 - alpha_t) + row1 * alpha_t

    def sample(self, t_now: float):
        if self._trajectory is None:
            return {
                "velocity_sp": self._constant_velocity,
                "accel_sp": self._constant_accel,
                "yaw_rate_sp": self._constant_yaw_rate,
            }

        row = self._sample_trajectory(t_now)
        return {
            "velocity_sp": row[1:4],
            "accel_sp": row[4:7],
            "yaw_rate_sp": row[7],
        }


class PX4LikeController(Backend):
    """
    Torch backend with two PX4-style input modes:
    - accel mode: accel+yaw_rate -> attitude -> rates -> torque
    - velocity mode: velocity+yaw_rate -> acceleration -> attitude -> rates -> torque
    """

    def __init__(
        self,
        trajectory_file: Optional[str] = None,
        results_file: Optional[str] = None,
        input_mode: str = "accel",
        velocity_command=(0.0, 0.0, 0.0),
        accel_command=(0.0, 0.0, 0.0),
        yaw_rate_command: float = 0.0,
        mass: float = 1.5,
        gravity: float = 9.81,
        max_tilt_deg: float = 50.0,
        thrust_limits=(0.0, 35.0),
        velocity_p_gains=(4.0, 4.0, 6.0),
        velocity_i_gains=(0.2, 0.2, 0.3),
        velocity_d_gains=(0.0, 0.0, 0.0),
        velocity_integrator_limits=(2.0, 2.0, 2.0),
        velocity_accel_limits=(8.0, 6.0, 6.0),
        attitude_p_gains=(6.0, 6.0, 3.0),
        rate_p_gains=(0.20, 0.20, 0.10),
        rate_i_gains=(0.10, 0.10, 0.08),
        rate_d_gains=(0.004, 0.004, 0.002),
        rate_limits=(3.5, 3.5, 2.5),
        rate_integrator_limits=(1.0, 1.0, 0.8),
        torque_limits=(0.6, 0.6, 0.25),
        input_offset=(0.0, 0.0, 0.0, 0.0),
        input_scaling=(1000.0, 1000.0, 1000.0, 1000.0),
        zero_position_armed=(100.0, 100.0, 100.0, 100.0),
        control_range=(0.0, 1.0),
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        data: bool = False,
    ):
        super().__init__(config=None)

        if input_mode not in ("accel", "velocity"):
            raise ValueError(f"Unsupported input_mode '{input_mode}'. Expected 'accel' or 'velocity'.")
        self.input_mode = input_mode

        if device.startswith("cuda") and not torch.cuda.is_available():
            carb.log_warn(f"Requested device '{device}' is not available. Falling back to 'cpu'.")
            device = "cpu"

        self.device = torch.device(device)
        self.dtype = dtype
        self._eps = float(torch.finfo(dtype).eps)
        self.data = data

        self.input_ref = torch.zeros(4, device=self.device, dtype=self.dtype)
        self._hil_actuator_controls = torch.zeros(4, device=self.device, dtype=self.dtype)

        self.p = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.q = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
        self.v = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.a = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.w = torch.zeros(3, device=self.device, dtype=self.dtype)

        self._received_first_state = False
        self._yaw_initialized = False
        self._yaw_sp = torch.as_tensor(0.0, device=self.device, dtype=self.dtype)
        self._total_time = 0.0

        self._command_source = MulticopterCommandSource(
            trajectory_file=trajectory_file,
            velocity_command=velocity_command,
            accel_command=accel_command,
            yaw_rate_command=yaw_rate_command,
            device=self.device,
            dtype=self.dtype,
        )

        self._pipeline = PX4LikeMulticopterCascade(
            mass=mass,
            gravity=gravity,
            max_tilt_deg=max_tilt_deg,
            thrust_limits=thrust_limits,
            velocity_p_gains=velocity_p_gains,
            velocity_i_gains=velocity_i_gains,
            velocity_d_gains=velocity_d_gains,
            velocity_integrator_limits=velocity_integrator_limits,
            velocity_accel_limits=velocity_accel_limits,
            attitude_p_gains=attitude_p_gains,
            rate_p_gains=rate_p_gains,
            rate_i_gains=rate_i_gains,
            rate_d_gains=rate_d_gains,
            rate_limits=rate_limits,
            rate_integrator_limits=rate_integrator_limits,
            torque_limits=torque_limits,
            device=self.device,
            dtype=self.dtype,
        )

        self._hil_mapper = HilActuatorMapper(
            input_offset=input_offset,
            input_scaling=input_scaling,
            zero_position_armed=zero_position_armed,
            control_min=control_range[0],
            control_max=control_range[1],
            device=self.device,
            dtype=self.dtype,
        )

        self.results_file = results_file
        self.time_history = []
        self.velocity_sp_history = []
        self.accel_sp_history = []
        self.yaw_rate_sp_history = []
        self.attitude_sp_history = []
        self.rates_sp_history = []
        self.torque_sp_history = []
        self.hil_controls_history = []
        self.motor_omega_history = []

    @property
    def hil_actuator_controls(self) -> torch.Tensor:
        return self._hil_actuator_controls

    def _as_tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _as_numpy(self, tensor: torch.Tensor):
        return tensor.detach().cpu().numpy()

    def start(self):
        self.reset_statistics()

    def stop(self):
        if self.results_file is None or len(self.time_history) == 0:
            return

        statistics = {
            "time": np.asarray(self.time_history),
            "input_mode": np.asarray([self.input_mode] * len(self.time_history)),
            "velocity_sp": np.vstack(self.velocity_sp_history),
            "accel_sp": np.vstack(self.accel_sp_history),
            "yaw_rate_sp": np.asarray(self.yaw_rate_sp_history),
            "attitude_sp": np.vstack(self.attitude_sp_history),
            "rates_sp": np.vstack(self.rates_sp_history),
            "torque_sp": np.vstack(self.torque_sp_history),
            "hil_controls": np.vstack(self.hil_controls_history),
            "motor_omega": np.vstack(self.motor_omega_history),
        }
        np.savez(self.results_file, **statistics)
        carb.log_warn("PX4-like controller statistics saved to: " + self.results_file)
        self.reset_statistics()

    def reset_statistics(self):
        self._total_time = 0.0
        self._yaw_initialized = False
        self._yaw_sp = torch.as_tensor(0.0, device=self.device, dtype=self.dtype)

        self.time_history = []
        self.velocity_sp_history = []
        self.accel_sp_history = []
        self.yaw_rate_sp_history = []
        self.attitude_sp_history = []
        self.rates_sp_history = []
        self.torque_sp_history = []
        self.hil_controls_history = []
        self.motor_omega_history = []

        self.input_ref.zero_()
        self._hil_actuator_controls.zero_()

        self._pipeline.reset()
        self._command_source.reset()

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state: State):
        self.p = self._as_tensor(state.position)
        self.q = self._as_tensor(state.attitude)
        self.v = self._as_tensor(state.linear_velocity)
        self.a = self._as_tensor(state.linear_acceleration)
        self.w = self._as_tensor(state.angular_velocity)
        self._received_first_state = True

        if self.data:
            vehicle_name = self.vehicle.vehicle_name if self.vehicle is not None else "vehicle"
            print(
                f"[{vehicle_name}] "
                f"v=({float(self.v[0]): .4f}, {float(self.v[1]): .4f}, {float(self.v[2]): .4f}) m/s "
                f"a=({float(self.a[0]): .4f}, {float(self.a[1]): .4f}, {float(self.a[2]): .4f}) m/s^2"
            )

    def input_reference(self):
        return self.input_ref

    def update(self, dt: float):
        if not self._received_first_state:
            return

        safe_dt = max(float(dt), self._eps)
        self._total_time += safe_dt

        command = self._command_source.sample(self._total_time)
        vel_sp = self._as_tensor(command["velocity_sp"])
        accel_sp = self._as_tensor(command["accel_sp"])
        yaw_rate_sp = self._as_tensor(command["yaw_rate_sp"])

        if not self._yaw_initialized:
            self._yaw_sp = yaw_from_quaternion(self.q)
            self._yaw_initialized = True

        self._yaw_sp = self._yaw_sp + yaw_rate_sp * safe_dt

        if self.input_mode == "velocity":
            outputs = self._pipeline.step_from_velocity(
                attitude=self.q,
                body_rates=self.w,
                velocity=self.v,
                velocity_sp=vel_sp,
                yaw_sp=self._yaw_sp,
                yaw_rate_sp=yaw_rate_sp,
                dt=safe_dt,
                accel_ff=accel_sp,
            )
        else:
            outputs = self._pipeline.step_from_acceleration(
                attitude=self.q,
                body_rates=self.w,
                accel_sp=accel_sp,
                yaw_sp=self._yaw_sp,
                yaw_rate_sp=yaw_rate_sp,
                dt=safe_dt,
            )

        if self.vehicle:
            rotor_omega = self.vehicle.force_and_torques_to_velocities(outputs["thrust_sp"], outputs["torque_sp"])
            rotor_omega = self._as_tensor(rotor_omega)
            self._hil_actuator_controls = self._hil_mapper.motor_omega_to_hil_controls(rotor_omega)
            self.input_ref = self._hil_mapper.hil_controls_to_motor_omega(self._hil_actuator_controls)

        self.time_history.append(self._total_time)
        self.velocity_sp_history.append(self._as_numpy(vel_sp))
        self.accel_sp_history.append(self._as_numpy(outputs["accel_sp"]))
        self.yaw_rate_sp_history.append(float(yaw_rate_sp.item()))
        self.attitude_sp_history.append(self._as_numpy(outputs["attitude_sp"]))
        self.rates_sp_history.append(self._as_numpy(outputs["rates_sp"]))
        self.torque_sp_history.append(self._as_numpy(outputs["torque_sp"]))
        self.hil_controls_history.append(self._as_numpy(self._hil_actuator_controls))
        self.motor_omega_history.append(self._as_numpy(self.input_ref))

    def reset(self):
        self.reset_statistics()
