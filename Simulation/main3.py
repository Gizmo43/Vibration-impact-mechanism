import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm

# ============================================================
# PARAMETERS
# ============================================================

R = 0.22

mu = 0.1
lam = 1.0

a = mu * lam

# ============================================================
# BIFURCATION PARAMETERS
# ============================================================

p_start = 0.1
p_end = 1.1
p_step = 0.0002

N_transient = 200
N_keep = 1000

# ============================================================
# LOCAL ANALYTICAL SOLUTION
# ============================================================

def velocity_local(t, v0, tau0, p):
    """
    Velocity between impacts

    t = tau - tau0
    """

    return (
        a * (
            np.sin(tau0 + t)
            - np.sin(tau0)
        )
        - p * t
        + v0
    )


def position_local(t, x0, v0, tau0, p):
    """
    Coordinate between impacts

    t = tau - tau0
    """

    return (
        -a * (
            np.cos(tau0 + t)
            - np.cos(tau0)
        )
        + v0 * t
        - 0.5 * p * t**2
        - a * t * np.sin(tau0)
        + x0
    )

# ============================================================
# GAP FUNCTION
# ============================================================

def gap_function(t, x0, v0, tau0, p):

    tau = tau0 + t

    x = position_local(
        t,
        x0,
        v0,
        tau0,
        p
    )

    return x + mu * np.cos(tau)

# ============================================================
# IMPACT SEARCH
# ============================================================

def find_impact_time(
        x0,
        v0,
        tau0,
        p,
        t_max=100.0,
        dt=0.02):
    """
    Find next impact time
    """

    t1 = 1e-10

    g1 = gap_function(
        t1,
        x0,
        v0,
        tau0,
        p
    )

    t = t1 + dt

    while t <= t_max:

        g2 = gap_function(
            t,
            x0,
            v0,
            tau0,
            p
        )

        if g1 * g2 <= 0:

            try:

                t_impact = brentq(
                    lambda x: gap_function(
                        x,
                        x0,
                        v0,
                        tau0,
                        p
                    ),
                    t - dt,
                    t,
                    xtol=1e-12,
                    rtol=1e-12
                )

                return t_impact

            except ValueError:
                pass

        g1 = g2
        t += dt

    return None

# ============================================================
# SINGLE IMPACT STEP
# ============================================================

def do_impact_step(x0, v0, tau0, p):
    """
    Advance the system to the next impact.
    Returns (tau_new, x_new, v_plus) or None if no impact found.
    """

    t_impact = find_impact_time(x0, v0, tau0, p)

    if t_impact is None:
        return None

    tau_impact = tau0 + t_impact

    x_minus = position_local(
        t_impact,
        x0,
        v0,
        tau0,
        p
    )

    v_minus = velocity_local(
        t_impact,
        v0,
        tau0,
        p
    )

    v_plus = (
        -R * v_minus
        + (1 + R)
        * mu
        * np.sin(tau_impact)
    )

    return tau_impact, x_minus, v_plus

# ============================================================
# DATA ARRAYS
# ============================================================

p_vals = []
v_vals = []

# ============================================================
# MAIN PARAMETER LOOP
# ============================================================

p_values = np.arange(
    p_start,
    p_end,
    p_step
)

for p in tqdm(
        p_values,
        desc="Bifurcation"):

    # ========================================================
    # INITIAL CONDITIONS
    # ========================================================

    tau0 = 0.0

    # сразу после удара
    x0 = -mu * np.cos(tau0)

    # постударная скорость
    v0 = 0.5

    # ========================================================
    # TRANSIENT: N_transient ударов
    # ========================================================

    n_transient_done = 0

    while n_transient_done < N_transient:

        result = do_impact_step(x0, v0, tau0, p)

        if result is None:
            break

        tau0, x0, v0 = result

        n_transient_done += 1

    # ========================================================
    # RECORD: N_keep ударов
    # ========================================================

    velocities = []

    while len(velocities) < N_keep:

        result = do_impact_step(x0, v0, tau0, p)

        if result is None:
            break

        tau0, x0, v0 = result

        velocities.append(v0)

    # ========================================================
    # STORE
    # ========================================================

    p_vals.extend(
        [p] * len(velocities)
    )

    v_vals.extend(velocities)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 7))

plt.scatter(
    p_vals,
    v_vals,
    s=0.2,
    alpha=0.6
)

plt.xlabel('p', fontsize=14)

plt.ylabel(
    'Velocity after impact',
    fontsize=14
)

plt.title(
    'Bifurcation diagram',
    fontsize=15
)

plt.grid(True)

plt.tight_layout()

plt.show()