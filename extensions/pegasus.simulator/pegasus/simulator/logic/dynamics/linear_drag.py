"""
| File: linear_drag.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Description: Computes the forces that should actuate on a rigidbody affected by linear drag
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
"""
import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None
from pegasus.simulator.logic.dynamics.drag import Drag
from pegasus.simulator.logic.state import State

class LinearDrag(Drag):
    """
    Class that implements linear drag computations afftecting a rigid body. It inherits the Drag base class.
    """

    def __init__(self, drag_coefficients=[0.0, 0.0, 0.0]):
        """
        Receives as input the drag coefficients of the vehicle as a 3x1 vector of constants

        Args:
            drag_coefficients (list[float]): The constant linear drag coefficients to used to compute the total drag forces
            affecting the rigid body. The linear drag is given by diag(dx, dy, dz) * [v_x, v_y, v_z] where the velocities
            are expressed in the body frame of the rigid body (using the FRU frame convention).
        """

        # Initialize the base Drag class
        super().__init__()

        # The linear drag coefficients of the vehicle's body frame
        if torch is None:
            self._drag_coefficients = np.diag(drag_coefficients)
        else:
            self._drag_coefficients = torch.diag(torch.as_tensor(drag_coefficients, dtype=torch.float32))

        # The drag force to apply on the vehicle's body frame
        self._drag_force = [0.0, 0.0, 0.0]

    @property
    def drag(self):
        """The drag force to be applied on the body frame of the vehicle

        Returns:
            list: A list with len==3 containing the drag force to be applied on the rigid body according to a FLU body reference
            frame, expressed in Newton (N) [dx, dy, dz]
        """
        return self._drag_force

    def update(self, state: State, dt: float):
        """Method that updates the drag force to be applied on the body frame of the vehicle. The total drag force
        applied on the body reference frame (FLU convention) is given by diag(dx,dy,dz) * R' * v
        where v is the velocity of the vehicle expressed in the inertial frame and R' * v = velocity_body_frame

        Args:
            state (State): The current state of the vehicle.
             dt (float): The time elapsed between the previous and current function calls (s).

        Returns:
            list: A list with len==3 containing the drag force to be applied on the rigid body according to a FLU body reference
        """

        # Get the velocity of the vehicle expressed in the body frame of reference
        body_vel = state.linear_body_velocity

        # Compute the component of the drag force to be applied in the body frame
        if torch is None:
            self._drag_force = (-np.dot(self._drag_coefficients, body_vel)).tolist()
        else:
            body_vel_t = torch.as_tensor(body_vel, dtype=self._drag_coefficients.dtype, device=self._drag_coefficients.device)
            self._drag_force = (-(self._drag_coefficients @ body_vel_t)).detach().cpu().tolist()
        return self._drag_force
