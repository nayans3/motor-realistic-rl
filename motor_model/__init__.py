"""Electrical motor model for the Go1 joints. See motor.py and params.py."""

from motor_model.motor import (Effects, MotorInfo, MotorState, init_state,
                               motor_step, thermal_derate, torque_speed_curve)
from motor_model.params import DEFAULT_PARAMS, GEAR, MotorParams

__all__ = [
    "DEFAULT_PARAMS", "GEAR", "Effects", "MotorInfo", "MotorParams",
    "MotorState", "init_state", "motor_step", "thermal_derate",
    "torque_speed_curve",
]
