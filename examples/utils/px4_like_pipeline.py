#!/usr/bin/env python
"""
| File: px4_like_pipeline.py
| Description: Torch implementation of a PX4-like multirotor control pipeline after acceleration setpoint generation.
"""

import math

import torch


def _safe_norm(vec: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.linalg.norm(vec).clamp_min(eps)


def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / torch.linalg.norm(q).clamp_min(eps)


def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix."""
    q = quat_normalize(q)
    x, y, z, w = q.unbind()

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return torch.stack(
        (
            torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy))),
            torch.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx))),
            torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy))),
        )
    )


def rot_matrix_to_quat(rotation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = rotation.trace()
    one = torch.tensor(1.0, device=rotation.device, dtype=rotation.dtype)
    trace_value = float(trace.item())
    r00 = float(rotation[0, 0].item())
    r11 = float(rotation[1, 1].item())
    r22 = float(rotation[2, 2].item())

    if trace_value > 0.0:
        s = torch.sqrt(trace + one) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif r00 > r11 and r00 > r22:
        s = torch.sqrt(one + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]).clamp_min(eps) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif r11 > r22:
        s = torch.sqrt(one + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]).clamp_min(eps) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = torch.sqrt(one + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]).clamp_min(eps) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s

    quat = torch.stack((qx, qy, qz, qw))
    return quat_normalize(quat, eps=eps)


def yaw_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (ENU convention) from quaternion [qx, qy, qz, qw]."""
    q = quat_normalize(q)
    x, y, z, w = q.unbind()
    yaw_num = 2.0 * (w * z + x * y)
    yaw_den = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(yaw_num, yaw_den)


def vee(S: torch.Tensor) -> torch.Tensor:
    return torch.stack((-S[1, 2], S[0, 2], -S[0, 1]))


class AccelYawRateToAttitude:
    """
    Approximation of PX4's acceleration/yaw setpoint conversion into attitude and thrust setpoints.
    """

    def __init__(
        self,
        mass: float,
        gravity: float,
        max_tilt_deg: float,
        max_thrust: float,
        min_thrust: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self._eps = float(torch.finfo(dtype).eps)
        self.mass = _as_tensor(mass, device, dtype)
        self.gravity = _as_tensor(gravity, device, dtype)
        self.max_thrust = _as_tensor(max_thrust, device, dtype)
        self.min_thrust = _as_tensor(min_thrust, device, dtype)
        self._world_z = _as_tensor([0.0, 0.0, 1.0], device, dtype)
        self.max_tilt_rad = _as_tensor(math.radians(max_tilt_deg), device, dtype)
        self._cos_max_tilt = torch.cos(self.max_tilt_rad)
        self._sin_max_tilt = torch.sin(self.max_tilt_rad)

    def _limit_tilt(self, z_b_des: torch.Tensor) -> torch.Tensor:
        if float(z_b_des[2].item()) < float(self._cos_max_tilt.item()):
            xy = z_b_des[:2]
            xy_norm = _safe_norm(xy, self._eps)
            xy = xy * (self._sin_max_tilt / xy_norm)
            z_b_des = torch.stack((xy[0], xy[1], self._cos_max_tilt))
            z_b_des = z_b_des / _safe_norm(z_b_des, self._eps)
        return z_b_des

    def update(self, accel_sp: torch.Tensor, yaw_sp: torch.Tensor):
        accel_sp = _as_tensor(accel_sp, self.device, self.dtype)
        yaw_sp = _as_tensor(yaw_sp, self.device, self.dtype)

        force_sp = self.mass * (accel_sp + self.gravity * self._world_z)
        thrust_sp = _safe_norm(force_sp, self._eps)
        thrust_sp = torch.clamp(thrust_sp, min=self.min_thrust, max=self.max_thrust)

        z_b_des = force_sp / _safe_norm(force_sp, self._eps)
        z_b_des = self._limit_tilt(z_b_des)

        x_c_des = torch.stack((torch.cos(yaw_sp), torch.sin(yaw_sp), _as_tensor(0.0, self.device, self.dtype)))
        y_b_des = torch.cross(z_b_des, x_c_des, dim=0)
        if float(torch.linalg.norm(y_b_des).item()) < self._eps:
            y_b_des = torch.stack((-torch.sin(yaw_sp), torch.cos(yaw_sp), _as_tensor(0.0, self.device, self.dtype)))
        y_b_des = y_b_des / _safe_norm(y_b_des, self._eps)
        x_b_des = torch.cross(y_b_des, z_b_des, dim=0)
        x_b_des = x_b_des / _safe_norm(x_b_des, self._eps)

        rotation_sp = torch.stack((x_b_des, y_b_des, z_b_des), dim=1)
        attitude_sp = rot_matrix_to_quat(rotation_sp, eps=self._eps)
        return attitude_sp, rotation_sp, thrust_sp


class AttitudePController:
    """Approximation of PX4 multicopter attitude controller (attitude error -> body-rate setpoint)."""

    def __init__(self, gains, max_rates, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.gains = _as_tensor(gains, device, dtype)
        self.max_rates = _as_tensor(max_rates, device, dtype)

    def update(self, attitude: torch.Tensor, rotation_sp: torch.Tensor, yaw_rate_sp: torch.Tensor):
        attitude = _as_tensor(attitude, self.device, self.dtype)
        yaw_rate_sp = _as_tensor(yaw_rate_sp, self.device, self.dtype)

        rotation = quat_to_rot_matrix(attitude)
        attitude_error = 0.5 * vee((rotation_sp.T @ rotation) - (rotation.T @ rotation_sp))
        rates_sp = -(self.gains * attitude_error)
        rates_sp[2] = rates_sp[2] + yaw_rate_sp
        rates_sp = torch.clamp(rates_sp, min=-self.max_rates, max=self.max_rates)
        return rates_sp, attitude_error


class RatePIDController:
    """Approximation of PX4 multicopter body-rate PID controller (rate setpoint -> body torque setpoint)."""

    def __init__(
        self,
        p_gains,
        i_gains,
        d_gains,
        integrator_limit,
        torque_limit,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self._eps = float(torch.finfo(dtype).eps)
        self.p_gains = _as_tensor(p_gains, device, dtype)
        self.i_gains = _as_tensor(i_gains, device, dtype)
        self.d_gains = _as_tensor(d_gains, device, dtype)
        self.integrator_limit = _as_tensor(integrator_limit, device, dtype)
        self.torque_limit = _as_tensor(torque_limit, device, dtype)

        self.integral = torch.zeros(3, device=device, dtype=dtype)
        self.prev_rates = torch.zeros(3, device=device, dtype=dtype)
        self.initialized = False

    def reset(self):
        self.integral.zero_()
        self.prev_rates.zero_()
        self.initialized = False

    def update(self, rates_sp: torch.Tensor, rates: torch.Tensor, dt: float):
        rates_sp = _as_tensor(rates_sp, self.device, self.dtype)
        rates = _as_tensor(rates, self.device, self.dtype)

        safe_dt = max(float(dt), self._eps)
        if not self.initialized:
            self.prev_rates = rates
            self.initialized = True

        rates_dot = (rates - self.prev_rates) / safe_dt
        rate_error = rates_sp - rates

        self.integral = self.integral + (rate_error * safe_dt)
        self.integral = torch.clamp(self.integral, min=-self.integrator_limit, max=self.integrator_limit)

        torque_sp = self.p_gains * rate_error + self.i_gains * self.integral - self.d_gains * rates_dot
        torque_sp = torch.clamp(torque_sp, min=-self.torque_limit, max=self.torque_limit)

        self.prev_rates = rates
        return torque_sp, rate_error, rates_dot


class VelocityPIDController:
    """PX4-like velocity controller block (velocity setpoint -> acceleration setpoint)."""

    def __init__(
        self,
        p_gains,
        i_gains,
        d_gains,
        integrator_limit,
        accel_limit_xy: float,
        accel_limit_up: float,
        accel_limit_down: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self._eps = float(torch.finfo(dtype).eps)
        self.p_gains = _as_tensor(p_gains, device, dtype)
        self.i_gains = _as_tensor(i_gains, device, dtype)
        self.d_gains = _as_tensor(d_gains, device, dtype)
        self.integrator_limit = _as_tensor(integrator_limit, device, dtype)
        self.accel_limit_xy = _as_tensor(accel_limit_xy, device, dtype)
        self.accel_limit_up = _as_tensor(accel_limit_up, device, dtype)
        self.accel_limit_down = _as_tensor(accel_limit_down, device, dtype)

        self.integral = torch.zeros(3, device=device, dtype=dtype)
        self.prev_error = torch.zeros(3, device=device, dtype=dtype)
        self.initialized = False

    def reset(self):
        self.integral.zero_()
        self.prev_error.zero_()
        self.initialized = False

    def update(
        self,
        vel_sp: torch.Tensor,
        vel: torch.Tensor,
        dt: float,
        accel_ff: torch.Tensor | None = None,
    ):
        vel_sp = _as_tensor(vel_sp, self.device, self.dtype)
        vel = _as_tensor(vel, self.device, self.dtype)
        accel_ff = (
            _as_tensor([0.0, 0.0, 0.0], self.device, self.dtype)
            if accel_ff is None
            else _as_tensor(accel_ff, self.device, self.dtype)
        )

        safe_dt = max(float(dt), self._eps)
        vel_error = vel_sp - vel

        if not self.initialized:
            self.prev_error = vel_error
            self.initialized = True

        vel_error_dot = (vel_error - self.prev_error) / safe_dt

        self.integral = self.integral + vel_error * safe_dt
        self.integral = torch.clamp(self.integral, min=-self.integrator_limit, max=self.integrator_limit)

        accel_sp = accel_ff + self.p_gains * vel_error + self.i_gains * self.integral + self.d_gains * vel_error_dot

        accel_xy = accel_sp[:2]
        accel_xy_norm = torch.linalg.norm(accel_xy)
        if float(accel_xy_norm.item()) > float(self.accel_limit_xy.item()):
            accel_xy = accel_xy * (self.accel_limit_xy / accel_xy_norm.clamp_min(self._eps))
            accel_sp = torch.stack((accel_xy[0], accel_xy[1], accel_sp[2]))

        # ENU convention: +Z is up, so use asymmetric up/down acceleration limits.
        accel_sp[2] = torch.clamp(accel_sp[2], min=-self.accel_limit_down, max=self.accel_limit_up)

        self.prev_error = vel_error
        return accel_sp, vel_error, vel_error_dot


class HilActuatorMapper:
    """
    Converts between rotor angular velocity and HIL_ACTUATOR_CONTROLS-style motor commands.
    """

    def __init__(
        self,
        input_offset,
        input_scaling,
        zero_position_armed,
        control_min,
        control_max,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self.input_offset = _as_tensor(input_offset, device, dtype)
        self.input_scaling = _as_tensor(input_scaling, device, dtype)
        self.zero_position_armed = _as_tensor(zero_position_armed, device, dtype)
        self.control_min = _as_tensor(control_min, device, dtype)
        self.control_max = _as_tensor(control_max, device, dtype)

    def motor_omega_to_hil_controls(self, rotor_omega: torch.Tensor) -> torch.Tensor:
        rotor_omega = _as_tensor(rotor_omega, self.device, self.dtype)
        controls = (rotor_omega - self.zero_position_armed) / self.input_scaling - self.input_offset
        return torch.clamp(controls, min=self.control_min, max=self.control_max)

    def hil_controls_to_motor_omega(self, controls: torch.Tensor) -> torch.Tensor:
        controls = _as_tensor(controls, self.device, self.dtype)
        controls = torch.clamp(controls, min=self.control_min, max=self.control_max)
        return (controls + self.input_offset) * self.input_scaling + self.zero_position_armed


class PX4LikeMulticopterCascade:
    """
    PX4-like multicopter cascade in Torch:
    velocity (optional) -> acceleration -> attitude -> body-rate -> torque.
    """

    def __init__(
        self,
        mass: float,
        gravity: float,
        max_tilt_deg: float,
        thrust_limits,
        velocity_p_gains,
        velocity_i_gains,
        velocity_d_gains,
        velocity_integrator_limits,
        velocity_accel_limits,
        attitude_p_gains,
        rate_p_gains,
        rate_i_gains,
        rate_d_gains,
        rate_limits,
        rate_integrator_limits,
        torque_limits,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if len(velocity_accel_limits) == 2:
            accel_limit_xy = velocity_accel_limits[0]
            accel_limit_up = velocity_accel_limits[1]
            accel_limit_down = velocity_accel_limits[1]
        elif len(velocity_accel_limits) == 3:
            accel_limit_xy = velocity_accel_limits[0]
            accel_limit_up = velocity_accel_limits[1]
            accel_limit_down = velocity_accel_limits[2]
        else:
            raise ValueError(
                "velocity_accel_limits must be (xy, z) or (xy, up, down)"
            )

        self.velocity_controller = VelocityPIDController(
            p_gains=velocity_p_gains,
            i_gains=velocity_i_gains,
            d_gains=velocity_d_gains,
            integrator_limit=velocity_integrator_limits,
            accel_limit_xy=accel_limit_xy,
            accel_limit_up=accel_limit_up,
            accel_limit_down=accel_limit_down,
            device=device,
            dtype=dtype,
        )
        self.accel_to_attitude = AccelYawRateToAttitude(
            mass=mass,
            gravity=gravity,
            max_tilt_deg=max_tilt_deg,
            max_thrust=thrust_limits[1],
            min_thrust=thrust_limits[0],
            device=device,
            dtype=dtype,
        )
        self.attitude_controller = AttitudePController(
            gains=attitude_p_gains,
            max_rates=rate_limits,
            device=device,
            dtype=dtype,
        )
        self.rate_controller = RatePIDController(
            p_gains=rate_p_gains,
            i_gains=rate_i_gains,
            d_gains=rate_d_gains,
            integrator_limit=rate_integrator_limits,
            torque_limit=torque_limits,
            device=device,
            dtype=dtype,
        )

    def reset(self):
        self.velocity_controller.reset()
        self.rate_controller.reset()

    def step_from_acceleration(
        self,
        attitude: torch.Tensor,
        body_rates: torch.Tensor,
        accel_sp: torch.Tensor,
        yaw_sp: torch.Tensor,
        yaw_rate_sp: torch.Tensor,
        dt: float,
    ):
        attitude_sp, rotation_sp, thrust_sp = self.accel_to_attitude.update(accel_sp=accel_sp, yaw_sp=yaw_sp)
        rates_sp, attitude_error = self.attitude_controller.update(
            attitude=attitude, rotation_sp=rotation_sp, yaw_rate_sp=yaw_rate_sp
        )
        torque_sp, rates_error, rates_dot = self.rate_controller.update(rates_sp=rates_sp, rates=body_rates, dt=dt)
        return {
            "accel_sp": _as_tensor(accel_sp, attitude.device, attitude.dtype),
            "attitude_sp": attitude_sp,
            "rotation_sp": rotation_sp,
            "thrust_sp": thrust_sp,
            "rates_sp": rates_sp,
            "torque_sp": torque_sp,
            "attitude_error": attitude_error,
            "rates_error": rates_error,
            "rates_dot": rates_dot,
            "velocity_error": None,
            "velocity_error_dot": None,
        }

    def step_from_velocity(
        self,
        attitude: torch.Tensor,
        body_rates: torch.Tensor,
        velocity: torch.Tensor,
        velocity_sp: torch.Tensor,
        yaw_sp: torch.Tensor,
        yaw_rate_sp: torch.Tensor,
        dt: float,
        accel_ff: torch.Tensor | None = None,
    ):
        accel_sp, vel_error, vel_error_dot = self.velocity_controller.update(
            vel_sp=velocity_sp, vel=velocity, dt=dt, accel_ff=accel_ff
        )
        outputs = self.step_from_acceleration(
            attitude=attitude,
            body_rates=body_rates,
            accel_sp=accel_sp,
            yaw_sp=yaw_sp,
            yaw_rate_sp=yaw_rate_sp,
            dt=dt,
        )
        outputs["velocity_error"] = vel_error
        outputs["velocity_error_dot"] = vel_error_dot
        return outputs


# Backward-compatible alias for previous class name.
PX4LikePostAccelPipeline = PX4LikeMulticopterCascade
