"""Electrical motor model for the Go1's 12 joint motors, written in JAX.

Sits between the PD controller and the physics:

    PD controller -> torque request -> [this model] -> torque the joint gets

Each effect has its own on/off switch (`Effects`), so experiment 1 can turn
them on one at a time. With every switch off, the output equals the 'ideal'
simulator: the torque request, clipped at the published max torque.

Current is treated as instantaneous within each 4 ms physics step (the drive's
current loop settles far faster than that); `verify_current_model.py` checks
this assumption against a fine-grained simulation.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jp

from motor_model.params import DEFAULT_PARAMS, MotorParams


@dataclass(frozen=True)
class Effects:
    """On/off switches. Static (Python bools), so JAX compiles one version
    per combination and switched-off effects cost nothing."""

    back_emf: bool = False
    """Voltage limit: the faster a joint spins, the less current (and torque)
    the battery voltage can push through it. This is the torque-speed curve."""

    battery_sag: bool = False
    """Bus voltage drops as the robot draws power. Acts through the voltage
    limit, so it only has an effect when back_emf is also on."""

    thermal: bool = False
    """Windings heat up with current; the driver cuts current when hot."""

    @classmethod
    def all_on(cls) -> "Effects":
        return cls(back_emf=True, battery_sag=True, thermal=True)


class MotorState(NamedTuple):
    v_bus: jp.ndarray  # battery bus voltage [V], scalar
    temp: jp.ndarray   # winding temperature per joint [deg C], shape (12,)


class MotorInfo(NamedTuple):
    current: jp.ndarray      # motor current per joint [A]
    power: jp.ndarray        # electrical power per joint [W] (negative = regen)
    i_limit: jp.ndarray      # current limit in force per joint [A]


def init_state(params: MotorParams = DEFAULT_PARAMS,
               temp0: float | jp.ndarray | None = None) -> MotorState:
    """Fresh battery and windings. `temp0` lets experiments start warm."""
    n = params.gear.shape[0]
    temp = jp.full((n,), params.t_amb if temp0 is None else temp0, jp.float32)
    return MotorState(v_bus=jp.float32(params.v_oc), temp=temp)


def thermal_derate(temp, params: MotorParams = DEFAULT_PARAMS):
    """Fraction of peak current allowed at a winding temperature (1 -> 0)."""
    span = params.t_derate_end - params.t_derate_start
    return jp.clip((params.t_derate_end - temp) / span, 0.0, 1.0)


def motor_step(tau_cmd, qvel, state: MotorState, dt: float,
               effects: Effects = Effects(),
               params: MotorParams = DEFAULT_PARAMS):
    """One physics step for all joints.

    Args:
      tau_cmd: requested joint torques from the PD controller [N·m], (12,)
      qvel: joint velocities [rad/s], (12,)
      state: battery voltage and winding temperatures
      dt: physics step [s] (0.004 for the Go1 task)
      effects: which effects are switched on
      params: motor constants

    Returns:
      (tau, new_state, info): delivered joint torques [N·m], next state, and
      per-joint current/power for logging.
    """
    gear = jp.asarray(params.gear)
    kt = params.kt * gear                  # joint-referred torque constant
    ke = params.ke * gear                  # joint-referred back-EMF constant
    tau_max = params.tau_max * gear        # 23.7 N·m, or 35.55 at the knees

    # 1. Torque request -> current request.
    current = tau_cmd / kt

    # 2. Current limit (peak current, lowered when the windings are hot).
    i_limit = jp.full_like(current, params.i_max)
    if effects.thermal:
        i_limit = i_limit * thermal_derate(state.temp, params)
    current = jp.clip(current, -i_limit, i_limit)

    # 3. Voltage limit: the driver can only apply -V_bus..+V_bus across the
    #    winding, and back-EMF (ke * speed) eats into that. Steady state:
    #    V = R*i + ke*qvel  ->  i must lie in [(-V - emf)/R, (V - emf)/R].
    emf = ke * qvel
    if effects.back_emf:
        v = state.v_bus if effects.battery_sag else jp.float32(params.v_oc)
        current = jp.clip(current, (-v - emf) / params.r, (v - emf) / params.r)

    # 4. Current -> torque, with the published torque cap (same as ideal sim).
    tau = jp.clip(kt * current, -tau_max, tau_max)
    current = tau / kt

    # 5. Electrical power drawn per joint: copper loss + mechanical power.
    power = params.r * current**2 + current * emf

    # 6. Battery sag: next step's bus voltage from this step's total draw.
    v_bus = state.v_bus
    if effects.battery_sag:
        i_bus = jp.sum(power) / state.v_bus
        v_bus = jp.clip(params.v_oc - params.r_batt * i_bus,
                        params.v_min, params.v_oc + 2.0)

    # 7. Winding temperature: copper loss heats, convection cools.
    temp = state.temp
    if effects.thermal:
        heat_in = params.r * current**2
        heat_out = (state.temp - params.t_amb) / params.r_th
        temp = state.temp + dt * (heat_in - heat_out) / params.c_th

    return tau, MotorState(v_bus=v_bus, temp=temp), MotorInfo(current, power, i_limit)


def torque_speed_curve(qvel, v_bus: float | None = None,
                       params: MotorParams = DEFAULT_PARAMS, knee: bool = False):
    """Max torque available at a joint speed (for plots and sanity checks)."""
    g = params.gear.max() if knee else 1.0
    kt, ke = params.kt * g, params.ke * g
    v = params.v_oc if v_bus is None else v_bus
    i_avail = jp.clip((v - ke * jp.abs(qvel)) / params.r, 0.0, params.i_max)
    return jp.minimum(kt * i_avail, params.tau_max * g)
