"""Motor parameters for the Unitree Go1 joint motor (GO-M8010-6).

Every value is either PUBLISHED (from Unitree's spec sheet) or ESTIMATED
(Unitree doesn't publish it). Estimates say how they were chosen, so they can
be defended or swept in experiment 3.

Units: SI throughout. All motor constants are referred to the motor's OUTPUT
shaft (after its built-in 6.33:1 gearbox), because that's how Unitree quotes
them. The knee joints add a further 1.5:1 reduction (handled by `gear`).

Sources:
  - Unitree GO-M8010-6 product page (max torque, torque constant, max speed,
    gear ratio, voltage range, max current): https://www.unitree.com/go1/motor
  - Unitree Go1 product page (knee joint = 1.5x ratio, 35.55 N·m):
    https://www.unitree.com/go1
  - MuJoCo Playground Go1 model (actuator torque caps ±23.7 / ±35.55 N·m).
"""

from dataclasses import dataclass, field

import numpy as np

# Joint order in the MuJoCo Go1 model: for each leg (FR, FL, RR, RL):
# hip (abduction), thigh, calf (knee). Knees are every third joint.
KNEE_GEAR = 1.5  # PUBLISHED: knee joint uses a 1.5x extra reduction.
GEAR = np.array([1.0, 1.0, KNEE_GEAR] * 4, dtype=np.float32)


@dataclass(frozen=True)
class MotorParams:
    # --- Motor, referred to its output shaft ---------------------------------
    kt: float = 0.63895
    """Torque constant [N·m/A]. PUBLISHED."""

    ke: float = 0.63895
    """Back-EMF constant [V/(rad/s)]. DERIVED: equal to kt in SI units for a
    DC-equivalent motor model (energy conservation). Gives an ideal no-load
    speed of 24 V / 0.639 = 37.6 rad/s, consistent with the published 30 rad/s
    max operating speed at 24 V."""

    r: float = 0.30
    """Winding resistance, DC-equivalent [ohm]. ESTIMATED: typical phase
    resistance for an ~80 mm, ~0.5 kg robot-joint motor is 0.2-0.4 ohm. With
    0.3 ohm the motor can still deliver about 10 N·m at its 30 rad/s max speed,
    which matches a usable top speed. Swept in experiment 3."""

    l: float = 0.3e-3
    """Winding inductance [H]. ESTIMATED: 0.1-1 mH is typical for this motor
    size. Only used by the current-model verification test, because the
    training model treats current as instantaneous (see verify_current_model)."""

    i_max: float = 40.0
    """Peak current [A]. PUBLISHED. Note kt * i_max = 25.6 N·m, just above the
    23.7 N·m torque cap, so the current limit only matters once thermal
    derating lowers it."""

    tau_max: float = 23.7
    """Peak output torque [N·m]. PUBLISHED. Same cap as the 'ideal' simulator."""

    # --- Battery --------------------------------------------------------------
    v_oc: float = 24.0
    """Battery open-circuit voltage [V]. PUBLISHED: motor rated for 12-30 V,
    24 V recommended; Go1 runs on a 24 V-class pack. Lower values (a draining
    battery) are swept in experiment 3."""

    r_batt: float = 0.10
    """Battery + wiring internal resistance [ohm]. ESTIMATED: typical for a
    ~6 Ah lithium-ion pack plus harness."""

    v_min: float = 12.0
    """Lowest bus voltage allowed [V]. PUBLISHED: bottom of the motor's 12-30 V
    operating range."""

    # --- Thermal (first-order model) -----------------------------------------
    r_th: float = 2.0
    """Winding-to-ambient thermal resistance [K/W]. ESTIMATED."""

    c_th: float = 60.0
    """Winding thermal capacitance [J/K]. ESTIMATED. With r_th this gives a
    2-minute thermal time constant, typical for a small motor without forced
    cooling."""

    t_amb: float = 25.0
    """Ambient temperature [deg C]. Assumed."""

    t_derate_start: float = 70.0
    """Winding temperature where the driver starts cutting current [deg C].
    ESTIMATED (typical protection setting)."""

    t_derate_end: float = 100.0
    """Winding temperature where current is cut to zero [deg C]. ESTIMATED."""

    # --- Per-joint gearing ----------------------------------------------------
    gear: np.ndarray = field(default_factory=lambda: GEAR.copy())
    """Extra reduction after the motor output, per joint (1.5 for knees)."""


DEFAULT_PARAMS = MotorParams()
