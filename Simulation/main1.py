import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ============================================================
# SYSTEM PARAMETERS
# ============================================================

R = 0.3
mu = 0.1
lam = 0.0
p = 0.1

a = mu * lam

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

tau_start = 0.0
tau_end = 200.0

# ============================================================
# ANALYTICAL SOLUTION
# ============================================================

def velocity(tau, tau0, v0):
    """
    Analytical solution for velocity
    """

    return (
        a * (np.sin(tau) - np.sin(tau0))
        - p * (tau - tau0)
        + v0
    )


def position(tau, tau0, x0, v0):
    """
    Analytical solution for coordinate
    """

    dt = tau - tau0

    return (
        -a * (np.cos(tau) - np.cos(tau0))
        + v0 * dt
        - 0.5 * p * dt**2
        - a * dt * np.sin(tau0)
        + x0
    )


# ============================================================
# IMPACT FUNCTION
# ============================================================

def impact_function(tau, tau0, x0, v0):
    """
    Impact condition:
        x + mu*cos(tau) = 0
    """

    return position(tau, tau0, x0, v0) + mu * np.cos(tau)


# ============================================================
# SEARCH NEXT IMPACT
# ============================================================

def find_next_impact(
        tau0,
        x0,
        v0,
        search_max=50.0,
        scan_step=0.02):
    """
    Search next impact using root finding
    """

    t_left = tau0 + 1e-6

    f_left = impact_function(
        t_left,
        tau0,
        x0,
        v0
    )

    t = t_left + scan_step

    while t <= tau0 + search_max:

        f_right = impact_function(
            t,
            tau0,
            x0,
            v0
        )

        # Root detected
        if f_left * f_right <= 0:

            tau_impact = brentq(
                lambda x: impact_function(
                    x,
                    tau0,
                    x0,
                    v0
                ),
                t - scan_step,
                t,
                xtol=1e-12,
                rtol=1e-12
            )

            return tau_impact

        f_left = f_right
        t += scan_step

    return None


# ============================================================
# INITIAL CONDITIONS
# ============================================================

tau = tau_start

x0 = 0.0
v0 = 0.0

# ============================================================
# DATA ARRAYS
# ============================================================

taus = []
xs = []

# ============================================================
# MAIN LOOP
# ============================================================

while tau < tau_end:

    # ========================================================
    # FIND NEXT IMPACT
    # ========================================================

    tau_impact = find_next_impact(
        tau,
        x0,
        v0
    )

    # No more impacts
    if tau_impact is None:
        tau_impact = tau_end

    # ========================================================
    # BUILD TRAJECTORY BETWEEN IMPACTS
    # ========================================================

    tau_segment = np.linspace(
        tau,
        min(tau_impact, tau_end),
        400
    )

    x_segment = position(
        tau_segment,
        tau,
        x0,
        v0
    )

    taus.extend(tau_segment)
    xs.extend(x_segment)

    # End simulation
    if tau_impact >= tau_end:
        break

    # ========================================================
    # VELOCITY BEFORE IMPACT
    # ========================================================

    v_minus = velocity(
        tau_impact,
        tau,
        v0
    )

    # ========================================================
    # IMPACT LAW
    # ========================================================

    v_plus = (
        -R * v_minus
        + (1 + R) * mu * np.sin(tau_impact)
    )

    # ========================================================
    # NEW INITIAL CONDITIONS
    # ========================================================

    tau = tau_impact + 1e-10

    x0 = -mu * np.cos(tau)
    v0 = v_plus


# ============================================================
# OSCILLOGRAM
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    taus,
    xs,
    linewidth=1.0
)

# Obstacle trajectory
taus_wall = np.linspace(tau_start, tau_end, 5000)
x_wall = -mu * np.cos(taus_wall)

plt.plot(
    taus_wall,
    x_wall,
    linestyle='--',
    linewidth=1.0
)

plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel(r'$x(\tau)$', fontsize=14)

plt.title(
    'Осциллограмма виброударного механизма',
    fontsize=15
)

plt.grid(True)

plt.tight_layout()
plt.show()