"""Check the 'instantaneous current' assumption.

The training model (motor.py) assumes the motor current reaches its target
within each 4 ms physics step. In reality the motor driver runs a fast current
loop: it adjusts the winding voltage thousands of times a second to push the
current toward the target, and the winding's inductance (L) slows how fast the
current can change.

This script simulates one motor both ways while it follows a walking-like
torque and speed profile:
  - FINE:    the real dynamics at 5 µs resolution: L di/dt = V - R i - ke w,
             with a PI current controller (1 kHz bandwidth) and the driver's
             voltage limit (it can't apply more than the battery voltage).
  - INSTANT: motor.py's model, one value per 4 ms step.
Then it compares the average torque each step delivers.

Because L isn't published, it repeats this for L = 0.1 to 1.0 mH.
Run it:  python -m motor_model.verify_current_model
"""

import numpy as np

from motor_model.motor import Effects, init_state, motor_step
from motor_model.params import DEFAULT_PARAMS

SIM_DT = 0.004        # Go1 physics step [s]
FINE_DT = 5e-6        # fine simulation step [s]
DURATION = 2.0        # seconds of walking-like motion
LOOP_BW_HZ = 1000.0   # current-loop bandwidth of the motor driver (typical)
L_VALUES = (0.1e-3, 0.3e-3, 0.5e-3, 1.0e-3)


def profiles(n_steps: int):
    """Walking-like joint speed and torque request, one value per 4 ms step.

    Speed swings +-15 rad/s at a 2 Hz gait. Torque has a smooth swing plus
    +-6 N·m jumps at foot contacts: the abrupt changes are the hard case for
    the instant-current assumption.
    """
    t = np.arange(n_steps) * SIM_DT
    qvel = 15.0 * np.sin(2 * np.pi * 2.0 * t)
    tau_cmd = 10.0 * np.sin(2 * np.pi * 2.0 * t + 1.0) + 6.0 * np.sign(
        np.sin(2 * np.pi * 4.0 * t))
    return qvel, tau_cmd


def instant_torques(qvel, tau_cmd, params=DEFAULT_PARAMS):
    """motor.py's model (back-EMF on, fixed 24 V), for joint 0 (a hip/thigh)."""
    out = np.zeros_like(tau_cmd)
    effects = Effects(back_emf=True)
    state = init_state(params)
    zeros = np.zeros(12, np.float32)
    for k in range(len(tau_cmd)):
        tc, qv = zeros.copy(), zeros.copy()
        tc[0], qv[0] = tau_cmd[k], qvel[k]
        tau, state, _ = motor_step(tc, qv, state, SIM_DT, effects, params)
        out[k] = float(tau[0])
    return out


def fine_torques(qvel, tau_cmd, l_values, params=DEFAULT_PARAMS):
    """Fine-grained current-loop simulation, all L values at once.

    Returns the average torque over each 4 ms step, shape (len(l_values), n).
    """
    p = params
    l = np.asarray(l_values, dtype=np.float64)
    wc = 2 * np.pi * LOOP_BW_HZ
    kp, ki = l * wc, p.r * wc            # standard PI tuning for an RL load
    n_sub = int(round(SIM_DT / FINE_DT))

    i = np.zeros_like(l)                 # winding current
    integ = np.zeros_like(l)             # PI integrator
    out = np.zeros((len(l), len(tau_cmd)))
    for k in range(len(tau_cmd)):
        i_ref = np.clip(tau_cmd[k] / p.kt, -p.i_max, p.i_max)
        w_next = qvel[k + 1] if k + 1 < len(qvel) else qvel[k]
        acc = np.zeros_like(l)
        for s in range(n_sub):
            w = qvel[k] + (w_next - qvel[k]) * s / n_sub   # speed varies smoothly
            emf = p.ke * w
            err = i_ref - i
            v_cmd = kp * err + ki * integ + emf          # PI + back-EMF feedforward
            v = np.clip(v_cmd, -p.v_oc, p.v_oc)          # driver voltage limit
            integ += np.where(v == v_cmd, err, 0.0) * FINE_DT   # anti-windup
            i += (v - p.r * i - emf) / l * FINE_DT
            acc += i
        tau = p.kt * acc / n_sub
        out[:, k] = np.clip(tau, -p.tau_max, p.tau_max)
    return out


def compare(l_values=L_VALUES, duration=DURATION, params=DEFAULT_PARAMS):
    """Relative RMS torque error of the instant model vs the fine model."""
    n = int(round(duration / SIM_DT))
    qvel, tau_cmd = profiles(n)
    inst = instant_torques(qvel, tau_cmd, params)
    fine = fine_torques(qvel, tau_cmd, l_values, params)
    rms = np.sqrt(np.mean(fine**2, axis=1))
    err = np.sqrt(np.mean((fine - inst) ** 2, axis=1)) / rms
    max_err = np.max(np.abs(fine - inst), axis=1)
    return {float(L): (float(e), float(m)) for L, e, m in zip(l_values, err, max_err)}


if __name__ == "__main__":
    print("Instant-current model vs fine current-loop simulation")
    print(f"({DURATION:.0f} s of walking-like motion, {LOOP_BW_HZ:.0f} Hz current loop)\n")
    print(f"{'L (mH)':>8} {'RMS error':>10} {'worst step':>12}")
    for L, (e, m) in compare().items():
        print(f"{L * 1e3:8.1f} {e * 100:9.1f}% {m:9.2f} N·m")
