#!/usr/bin/env python
"""
| File: nonlinear_controller.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS. In this controller, we
provide a quick way of following a given trajectory specified in csv files or track an hard-coded trajectory based
on exponentials! NOTE: This is just an example, to demonstrate the potential of the API. A much more flexible solution
can be achieved
"""

# Imports to be able to log to the terminal with fancy colors
import carb

# Imports from the Pegasus library
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends import Backend

# Auxiliary modules
import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "examples/utils/nonlinear_controller.py requires PyTorch. "
        "Install torch in the Isaac Sim Python environment to use this controller."
    ) from exc


class NonlinearController(Backend):
    """A nonlinear controller class. It implements a nonlinear controller that allows a vehicle to track
    aggressive trajectories. This controlers is well described in the papers

    [1] J. Pinto, B. J. Guerreiro and R. Cunha, "Planning Parcel Relay Manoeuvres for Quadrotors,"
    2021 International Conference on Unmanned Aircraft Systems (ICUAS), Athens, Greece, 2021,
    pp. 137-145, doi: 10.1109/ICUAS51884.2021.9476757.
    [2] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors,"
    2011 IEEE International Conference on Robotics and Automation, Shanghai, China, 2011,
    pp. 2520-2525, doi: 10.1109/ICRA.2011.5980409.
    """

    def __init__(
        self,
        trajectory_file: str = None,
        results_file: str = None,
        reverse=False,
        Kp=[10.0, 10.0, 10.0],
        Kd=[8.5, 8.5, 8.5],
        Ki=[1.50, 1.50, 1.50],
        Kr=[3.5, 3.5, 3.5],
        Kw=[0.5, 0.5, 0.5],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):

        super().__init__(config=None)

        self.device = torch.device(device)
        self.dtype = dtype
        self._eps = torch.finfo(dtype).eps

        # The current rotor references [rad/s]
        self.input_ref = torch.zeros(4, device=self.device, dtype=self.dtype)

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.q = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
        self.w = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.a = torch.zeros(3, device=self.device, dtype=self.dtype)

        # Define the control gains matrix for the outer-loop
        self.Kp = torch.diag(self._as_tensor(Kp))
        self.Kd = torch.diag(self._as_tensor(Kd))
        self.Ki = torch.diag(self._as_tensor(Ki))
        self.Kr = torch.diag(self._as_tensor(Kr))
        self.Kw = torch.diag(self._as_tensor(Kw))

        self.int = torch.zeros(3, device=self.device, dtype=self.dtype)

        # Define the dynamic parameters for the vehicle
        self.m = 1.50  # Mass in Kg
        self.g = 9.81  # The gravity acceleration ms^-2
        self._gravity_term = self._as_tensor([0.0, 0.0, self.m * self.g])

        # Read the target trajectory from a CSV file inside the trajectories directory
        # if a trajectory is provided. Otherwise, just perform the hard-coded trajectory provided with this controller
        self.index = 0
        if trajectory_file is not None:
            self.trajectory = self.read_trajectory_from_csv(trajectory_file)
            self.max_index = int(self.trajectory.shape[0])
            self.total_time = 0.0
        else:
            # Set the initial time for starting when using the built-in trajectory (the time is also used in this case
            # as the parametric value)
            self.total_time = -5.0
            # Signal that we will not used a received trajectory
            self.trajectory = None
            self.max_index = 1

        self.reverse = reverse

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.reveived_first_state = False

        # Lists used for analysing performance statistics
        self.results_files = results_file
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.atittude_error_over_time = []
        self.attitude_rate_error_over_time = []

    def _as_tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _as_numpy(self, value: torch.Tensor):
        return value.detach().cpu().numpy()

    @staticmethod
    def _quat_to_rot_matrix(q: torch.Tensor):
        """Convert quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix."""
        q = q / torch.linalg.norm(q).clamp_min(1e-12)
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

    def read_trajectory_from_csv(self, file_name: str):
        """Auxiliar method used to read the desired trajectory from a CSV file

        Args:
            file_name (str): A string with the name of the trajectory inside the trajectories directory

        Returns:
            torch.Tensor: A matrix with the trajectory desired states over time.
        """

        data = np.flip(np.genfromtxt(file_name, delimiter=","), axis=0).copy()
        return self._as_tensor(data)

    def start(self):
        """
        Reset the control and trajectory index
        """
        self.reset_statistics()

    def stop(self):
        """
        Stopping the controller. Saving the statistics data for plotting later
        """

        # Check if we should save the statistics to some file or not
        if self.results_files is None:
            return

        statistics = {}
        statistics["time"] = np.array(self.time_vector)
        statistics["p"] = np.vstack(self.position_over_time)
        statistics["desired_p"] = np.vstack(self.desired_position_over_time)
        statistics["ep"] = np.vstack(self.position_error_over_time)
        statistics["ev"] = np.vstack(self.velocity_error_over_time)
        statistics["er"] = np.vstack(self.atittude_error_over_time)
        statistics["ew"] = np.vstack(self.attitude_rate_error_over_time)
        np.savez(self.results_files, **statistics)
        carb.log_warn("Statistics saved to: " + self.results_files)

        self.reset_statistics()

    def update_sensor(self, sensor_type: str, data):
        """
        Do nothing. For now ignore all the sensor data and just use the state directly for demonstration purposes.
        This is a callback that is called at every physics step.

        Args:
            sensor_type (str): The name of the sensor providing the data
            data (dict): A dictionary that contains the data produced by the sensor
        """
        pass

    def update_state(self, state: State):
        """
        Method that updates the current state of the vehicle. This is a callback that is called at every physics step

        Args:
            state (State): The current state of the vehicle.
        """
        self.p = self._as_tensor(state.position)
        self.q = self._as_tensor(state.attitude)
        self.w = self._as_tensor(state.angular_velocity)
        self.v = self._as_tensor(state.linear_velocity)

        self.reveived_first_state = True

    def input_reference(self):
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            A list with the target angular velocities for each individual rotor of the vehicle
        """
        return self.input_ref

    def update(self, dt: float):
        """Method that implements the nonlinear control law and updates the target angular velocities for each rotor.
        This method will be called by the simulation on every physics step

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if self.reveived_first_state is False:
            return

        # -------------------------------------------------
        # Update the references for the controller to track
        # -------------------------------------------------
        self.total_time += dt

        # Check if we need to update to the next trajectory index
        if (
            self.trajectory is not None
            and self.index < self.max_index - 1
            and self.total_time >= float(self.trajectory[self.index + 1, 0].item())
        ):
            self.index += 1

        # Update using an external trajectory
        if self.trajectory is not None:
            # the target positions [m], velocity [m/s], accelerations [m/s^2], jerk [m/s^3], yaw-angle [rad], yaw-rate [rad/s]
            row = self.trajectory[self.index]
            p_ref = row[1:4]
            v_ref = row[4:7]
            a_ref = row[7:10]
            j_ref = row[10:13]
            yaw_ref = row[13]
            yaw_rate_ref = row[14]
        # Or update the reference using the built-in trajectory
        else:
            s = 0.6
            p_ref = self.pd(self.total_time, s, self.reverse)
            v_ref = self.d_pd(self.total_time, s, self.reverse)
            a_ref = self.dd_pd(self.total_time, s, self.reverse)
            j_ref = self.ddd_pd(self.total_time, s, self.reverse)
            yaw_ref = self.yaw_d(self.total_time, s)
            yaw_rate_ref = self.d_yaw_d(self.total_time, s)

        # -------------------------------------------------
        # Start the controller implementation
        # -------------------------------------------------

        # Compute the tracking errors
        ep = self.p - p_ref
        ev = self.v - v_ref
        self.int = self.int + (ep * dt)
        ei = self.int

        # Compute F_des term
        F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + self._gravity_term + (self.m * a_ref)

        # Get the current axis Z_B (given by the last column of the rotation matrix)
        R = self._quat_to_rot_matrix(self.q)
        Z_B = R[:, 2]

        # Get the desired total thrust in Z_B direction (u_1)
        u_1 = torch.dot(F_des, Z_B)

        # Compute the desired body-frame axis Z_b
        Z_b_des = F_des / torch.linalg.norm(F_des).clamp_min(self._eps)

        # Compute X_C_des
        X_c_des = torch.stack((torch.cos(yaw_ref), torch.sin(yaw_ref), self._as_tensor(0.0)))

        # Compute Y_b_des
        Z_b_cross_X_c = torch.cross(Z_b_des, X_c_des, dim=0)
        Y_b_des = Z_b_cross_X_c / torch.linalg.norm(Z_b_cross_X_c).clamp_min(self._eps)

        # Compute X_b_des
        X_b_des = torch.cross(Y_b_des, Z_b_des, dim=0)

        # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
        R_des = torch.stack((X_b_des, Y_b_des, Z_b_des), dim=1)

        # Compute the rotation error
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

        # Compute an approximation of the current vehicle acceleration in the inertial frame (since we cannot measure it directly)
        self.a = (u_1 * Z_B) / self.m - self._as_tensor([0.0, 0.0, self.g])

        # Compute the desired angular velocity by projecting the angular velocity in the Xb-Yb plane
        # projection of angular velocity on xB − yB plane
        # see eqn (7) from [2].
        u_1_safe = torch.sign(u_1) * torch.abs(u_1).clamp_min(self._eps)
        hw = (self.m / u_1_safe) * (j_ref - torch.dot(Z_b_des, j_ref) * Z_b_des)

        # desired angular velocity
        w_des = torch.stack(
            (
                -torch.dot(hw, Y_b_des),
                torch.dot(hw, X_b_des),
                yaw_rate_ref * Z_b_des[2],
            )
        )

        # Compute the angular velocity error
        e_w = self.w - w_des

        # Compute the torques to apply on the rigid body
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

        # Use the allocation matrix provided by the Multirotor vehicle to convert the desired force and torque
        # to angular velocity [rad/s] references to give to each rotor
        if self.vehicle:
            ang_vel = self.vehicle.force_and_torques_to_velocities(u_1, tau)

            if isinstance(ang_vel, torch.Tensor):
                self.input_ref = ang_vel
            elif isinstance(ang_vel, np.ndarray):
                self.input_ref = self._as_tensor(ang_vel)
            else:
                self.input_ref = self._as_tensor(list(ang_vel))

        # ----------------------------
        # Statistics to save for later
        # ----------------------------
        self.time_vector.append(self.total_time)
        self.position_over_time.append(self._as_numpy(self.p))
        self.desired_position_over_time.append(self._as_numpy(p_ref))
        self.position_error_over_time.append(self._as_numpy(ep))
        self.velocity_error_over_time.append(self._as_numpy(ev))
        self.atittude_error_over_time.append(self._as_numpy(e_R))
        self.attitude_rate_error_over_time.append(self._as_numpy(e_w))

    @staticmethod
    def vee(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.

        Args:
            S (torch.Tensor): A matrix in so(3)
        """
        return torch.stack((-S[1, 2], S[0, 2], -S[0, 1]))

    def reset_statistics(self):

        self.index = 0
        self.int.zero_()

        # If we received an external trajectory, reset the time to 0.0
        if self.trajectory is not None:
            self.total_time = 0.0
        # if using the internal trajectory, make the parametric value start at -5.0
        else:
            self.total_time = -5.0

        # Reset the lists used for analysing performance statistics
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.atittude_error_over_time = []
        self.attitude_rate_error_over_time = []

    # ---------------------------------------------------
    # Definition of an exponential trajectory for example
    # This can be used as a reference if not trajectory file is passed
    # as an argument to the constructor of this class
    # ---------------------------------------------------

    def pd(self, t, s, reverse=False):
        """The desired position of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            torch.Tensor: A 3x1 tensor with the x, y ,z desired [m]
        """

        t = self._as_tensor(t)
        s = self._as_tensor(s)

        exp_term = torch.exp(-0.5 * torch.pow(t / s, 2.0))
        x = t
        z = (1.0 / s) * exp_term + 1.0
        y = (1.0 / s) * exp_term

        if reverse is True:
            y = -(1.0 / s) * exp_term + 4.5

        return torch.stack((x, y, z))

    def d_pd(self, t, s, reverse=False):
        """The desired velocity of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            torch.Tensor: A 3x1 tensor with the d_x, d_y ,d_z desired [m/s]
        """

        t = self._as_tensor(t)
        s = self._as_tensor(s)

        x = self._as_tensor(1.0)
        exp_term = torch.exp(-torch.pow(t, 2.0) / (2.0 * torch.pow(s, 2.0)))
        y = -(t * exp_term) / torch.pow(s, 3.0)
        z = -(t * exp_term) / torch.pow(s, 3.0)

        if reverse is True:
            y = (t * exp_term) / torch.pow(s, 3.0)

        return torch.stack((x, y, z))

    def dd_pd(self, t, s, reverse=False):
        """The desired acceleration of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            torch.Tensor: A 3x1 tensor with the dd_x, dd_y ,dd_z desired [m/s^2]
        """

        t = self._as_tensor(t)
        s = self._as_tensor(s)

        x = self._as_tensor(0.0)
        exp_term = torch.exp(-torch.pow(t, 2.0) / (2.0 * torch.pow(s, 2.0)))
        y = (torch.pow(t, 2.0) * exp_term) / torch.pow(s, 5.0) - exp_term / torch.pow(s, 3.0)
        z = (torch.pow(t, 2.0) * exp_term) / torch.pow(s, 5.0) - exp_term / torch.pow(s, 3.0)

        if reverse is True:
            y = exp_term / torch.pow(s, 3.0) - (torch.pow(t, 2.0) * exp_term) / torch.pow(s, 5.0)

        return torch.stack((x, y, z))

    def ddd_pd(self, t, s, reverse=False):
        """The desired jerk of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            torch.Tensor: A 3x1 tensor with the ddd_x, ddd_y ,ddd_z desired [m/s^3]
        """

        t = self._as_tensor(t)
        s = self._as_tensor(s)

        x = self._as_tensor(0.0)
        exp_term = torch.exp(-torch.pow(t, 2.0) / (2.0 * torch.pow(s, 2.0)))
        y = (3.0 * t * exp_term) / torch.pow(s, 5.0) - (torch.pow(t, 3.0) * exp_term) / torch.pow(s, 7.0)
        z = (3.0 * t * exp_term) / torch.pow(s, 5.0) - (torch.pow(t, 3.0) * exp_term) / torch.pow(s, 7.0)

        if reverse is True:
            y = (torch.pow(t, 3.0) * exp_term) / torch.pow(s, 7.0) - (3.0 * t * exp_term) / torch.pow(s, 5.0)

        return torch.stack((x, y, z))

    def yaw_d(self, t, s):
        """The desired yaw of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            torch.Tensor: A float tensor with the desired yaw in rad
        """
        return self._as_tensor(0.0)

    def d_yaw_d(self, t, s):
        """The desired yaw_rate of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            torch.Tensor: A float tensor with the desired yaw_rate in rad/s
        """
        return self._as_tensor(0.0)

    def reset(self):
        """
        Method that when implemented, should handle the reset of the vehicle simulation to its original state
        """
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """
        For this demo we do not care about graphical sensors such as camera, therefore we can ignore this callback
        """
        pass
