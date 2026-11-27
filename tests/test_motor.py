"""Unit tests for the motor model.

Run in VS Code's Testing tab, or:  python -m pytest tests
Each test checks one physical behavior we expect from a real motor.
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from motor_model import (DEFAULT_PARAMS, Effects, init_state, motor_step,
                         thermal_derate, torque_speed_curve)
from motor_model.verify_current_model import compare

P = DEFAULT_PARAMS
DT = 0.004
N = 12
KNEE = 2       # index of the first knee joint (FR_calf)
THIGH = 1      # index of the first thigh joint (FR_thigh)


def one_joint(value, joint=THIGH):
    x = np.zeros(N, np.float32)
    x[joint] = value
    return jp.asarray(x)


def step(tau, qvel, effects=Effects(), state=None):
    state = init_state() if state is None else state
    return motor_step(tau, qvel, state, DT, effects)


# --- With everything off, the model must match the 'ideal' simulator ----------

def test_all_off_matches_ideal_simulator():
    """Ideal sim = torque request clipped at ±23.7 N·m (±35.55 at knees)."""
    rng = np.random.default_rng(0)
    tau_cmd = jp.asarray(rng.uniform(-50, 50, N), jp.float32)
    qvel = jp.asarray(rng.uniform(-30, 30, N), jp.float32)
    tau, _, _ = step(tau_cmd, qvel)
    expected = np.clip(np.asarray(tau_cmd), -P.tau_max * P.gear, P.tau_max * P.gear)
    np.testing.assert_allclose(tau, expected, rtol=1e-5)


def test_knee_has_higher_torque_cap():
    tau, _, _ = step(one_joint(100.0, KNEE), jp.zeros(N))
    assert float(tau[KNEE]) == pytest.approx(35.55, rel=1e-4)


# --- Back-EMF: the torque-speed curve -------------------------------------------

def test_full_torque_at_stall():
    tau, _, _ = step(one_joint(23.7), jp.zeros(N), Effects(back_emf=True))
    assert float(tau[THIGH]) == pytest.approx(23.7, rel=1e-4)


def test_torque_drops_with_speed():
    """Pushing in the direction of motion, available torque falls with speed."""
    speeds = [0.0, 10.0, 20.0, 25.0, 30.0, 35.0]
    torques = []
    for w in speeds:
        tau, _, _ = step(one_joint(23.7), one_joint(w), Effects(back_emf=True))
        torques.append(float(tau[THIGH]))
    assert all(a >= b for a, b in zip(torques, torques[1:]))
    assert torques[0] > 20.0 and torques[-1] < 5.0


def test_no_torque_beyond_no_load_speed():
    """Above V/ke = 37.6 rad/s the motor can't push forward at all."""
    tau, _, _ = step(one_joint(23.7), one_joint(40.0), Effects(back_emf=True))
    assert float(tau[THIGH]) <= 0.0


def test_braking_is_not_limited_by_back_emf():
    """Pushing against the motion (braking), back-EMF helps, so full torque."""
    tau, _, _ = step(one_joint(-23.7), one_joint(25.0), Effects(back_emf=True))
    assert float(tau[THIGH]) == pytest.approx(-23.7, rel=1e-4)


def test_matches_torque_speed_curve_helper():
    for w in [0.0, 15.0, 25.0, 32.0]:
        tau, _, _ = step(one_joint(100.0), one_joint(w), Effects(back_emf=True))
        assert float(tau[THIGH]) == pytest.approx(float(torque_speed_curve(w)), rel=1e-4)


def test_knee_speed_limit_is_lower():
    """The 1.5x knee gear gives 1.5x the torque but 1/1.5 the top speed."""
    w = 25.0 / 1.5 * 1.2  # fast for a knee
    knee = float(torque_speed_curve(w, knee=True))
    full = float(torque_speed_curve(0.0, knee=True))
    assert knee < full


# --- Battery sag ---------------------------------------------------------------

def test_battery_sags_under_load():
    effects = Effects(back_emf=True, battery_sag=True)
    tau_cmd = jp.full(N, 20.0)
    qvel = jp.full(N, 10.0)             # all joints working hard, moving
    _, state, info = step(tau_cmd, qvel, effects)
    assert float(jp.sum(info.power)) > 0
    assert float(state.v_bus) < P.v_oc


def test_low_battery_reduces_torque_at_speed():
    effects = Effects(back_emf=True, battery_sag=True)
    full = init_state()
    low = full._replace(v_bus=jp.float32(16.0))
    tau_full, _, _ = step(one_joint(23.7), one_joint(20.0), effects, full)
    tau_low, _, _ = step(one_joint(23.7), one_joint(20.0), effects, low)
    assert float(tau_low[THIGH]) < float(tau_full[THIGH])


def test_battery_sag_needs_back_emf_to_matter():
    """Sag acts through the voltage limit; without it, torque is unchanged."""
    low = init_state()._replace(v_bus=jp.float32(12.0))
    tau, _, _ = step(one_joint(20.0), one_joint(20.0), Effects(battery_sag=True), low)
    assert float(tau[THIGH]) == pytest.approx(20.0, rel=1e-5)


# --- Thermal -------------------------------------------------------------------

def test_windings_heat_up_under_current():
    effects = Effects(thermal=True)
    state = init_state()
    for _ in range(250):                 # 1 s of hard holding torque
        _, state, _ = motor_step(one_joint(20.0), jp.zeros(N), state, DT, effects)
    assert float(state.temp[THIGH]) > P.t_amb + 1.0
    assert float(state.temp[0]) == pytest.approx(P.t_amb)   # idle joint stays cool


def test_hot_motor_is_derated():
    hot = init_state(temp0=90.0)          # between 70 (start) and 100 (zero)
    tau, _, info = step(one_joint(23.7), jp.zeros(N), Effects(thermal=True), hot)
    assert float(info.i_limit[THIGH]) == pytest.approx(P.i_max / 3, rel=1e-4)
    assert float(tau[THIGH]) < 10.0


def test_derate_curve_endpoints():
    assert float(thermal_derate(50.0)) == 1.0
    assert float(thermal_derate(100.0)) == 0.0


# --- JAX: must compile and run on a batch of robots ----------------------------

def test_jit_and_vmap():
    effects = Effects.all_on()
    f = jax.jit(jax.vmap(lambda t, w, s: motor_step(t, w, s, DT, effects)))
    batch = 8
    states = jax.vmap(lambda _: init_state())(jp.arange(batch))
    tau, new_states, _ = f(jp.ones((batch, N)) * 10.0, jp.ones((batch, N)), states)
    assert tau.shape == (batch, N)
    assert new_states.temp.shape == (batch, N)
    assert bool(jp.all(jp.isfinite(tau)))


# --- The instant-current assumption ---------------------------------------------

def test_instant_current_assumption_holds():
    """Instant current is within 5% of a 5 µs current-loop simulation, for
    every plausible inductance (0.1-1 mH). See verify_current_model.py."""
    for L, (rms_err, _) in compare().items():
        assert rms_err < 0.05, f"L={L * 1e3:.1f} mH: {rms_err:.1%} error"
