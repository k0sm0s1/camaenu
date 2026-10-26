"""
Basic quadruped walking gait for the elephant robot (Task #8).

Per-leg foot trajectory is generated as a target (x, z) offset from the
hip in the sagittal plane (x = fore/aft, z = up/down, both relative to
the leg's own stance neutral position) and converted to (hip, knee)
joint angles via the numeric IK solver in leg_ik.py -- NOT via ad-hoc
sinusoids directly on the joint angles, because (measured earlier) hip
angle alone does not hold height constant, so a joint-space sinusoid
would make the body bob up and down by several cm every step.

--- How the duty factor (DUTY=0.9) was arrived at, empirically -------

A CoM check (mj_forward + body-mass-weighted xipos) found the whole
62.7kg robot's centre of mass sits almost exactly at the geometric
centre of its rectangular 4-foot stance (CoM=(-0.011, 0.000), hip
corners at (+-0.22, +-0.185)). Removing any one foot for a swing phase
leaves a support triangle whose far edge is the rectangle's diagonal --
which passes almost exactly through a centred CoM, leaving only ~7mm of
real margin no matter which leg swings. A first attempt at DUTY=0.75
(25% of the cycle in swing) blew straight through that margin: the
robot fell over (tilt runs away past 170deg) within about 3 seconds
of walking starting, reproducibly, regardless of settle time, footfall
order, or which leg led (validate_walk.py history).

Tried and reverted: an active per-swing lateral weight-shift (biasing
all 4 feet's targets, via the abduction joint, toward the centroid of
whichever 3 feet would remain once a leg lifts -- a real, standard
static-gait technique, and the exact closed-form target for a
rectangular stance is (-+HIP_X/3, -+HIP_Y/3), opposite sign to the
removed corner). A first version (an always-on sin(4*pi*g)/sin(2*pi*g)
pair matched to the 4 swing events) fixed the fall-over but replaced it
with a huge, monotonic 0.8m sideways slide within 2 cycles, because
that signal doesn't return to the same value across any one leg's own
75%-of-cycle stance window and so quietly added a real unintended
sideways stride. A second version (a per-swing pulse, zero everywhere
except a sin(pi*v) ramp during the actually-swinging leg's own window,
verified to return to exactly zero at every leg's own stance
start/end) still produced essentially the same magnitude of runaway
sideways slide and fall-over, meaning the active weight-shift itself
was not a targeted-enough intervention (something about actively
driving the *stance* legs' abduction while they're bearing load appears
to induce real slip/momentum effects well beyond the small kinematic
target) -- not something to keep debugging indefinitely within a first
walking phase.

What worked for that first phase: dropping to a much higher duty
factor (0.9 -> each leg swings for only 10% of the cycle) with a small
stride and lift, and NO active weight-shift at all. This barely
touched the thin static margin for a much shorter fraction of the
cycle, and was verified (validate_walk.py) to hold tilt under ~2.5deg
with STRIDE=0.03/LIFT=0.02/CYCLE_T=3.0 -- but only made slow net
progress (~0.03 m/s on the current, post-mass-fix robot). This was the
single-leg-at-a-time CREEP gait; it has since been superseded by the
diagonal TROT gait below (Task #14, direct user request: "use two
diagonal legs walking at ones to move it faster"). Kept here for
history since the swing-shape math (per-leg sinusoidal lift + linear
stride sweep) is unchanged -- only PHASE_OFFSET and the four tuned
constants below changed.

--- Diagonal trot (Task #14), validated empirically -------------------

The flagged worry going in: lifting one leg from this rectangular
stance already only leaves ~7mm of CoM margin against the remaining
support triangle's far edge (that edge IS the stance rectangle's own
diagonal -- see above). Lifting a full diagonal PAIR at once removes a
whole support triangle, leaving only the OPPOSITE two feet -- a
support LINE, not an area, with (in principle) zero margin for any
CoM motion perpendicular to that line.

Tested directly (sweep_trot.py / sweep_trot2.py) rather than assumed
unsafe or assumed fine: it turned out to work well, and better than
expected. The likely reason: this stance's near-exact rectangular
symmetry means BOTH diagonals cross the same centre point the CoM
already sits at, so the remaining support line still passes almost
exactly under the CoM -- the theoretical zero-margin direction only
gets exercised by real dynamic effects (leg-swing momentum, actuator
overshoot), not by any static offset, and those turned out to be small
enough at this stride/speed for the existing PD leg actuators to
correct within each cycle. A 30-second sweep across duty factor
(0.80/0.85/0.90), stride (0.05/0.06), cycle time (1.2/1.5/1.5s) and
lift height found EVERY combination tried stayed well clear of
trouble (max tilt 4.3-6.7deg, nowhere near the 25+ deg regime that
caused real problems before) -- so the limiting factor here was net
lateral drift creeping in at lower duty factors (DUTY=0.80 drifted
-0.27m over 30s, worse than DUTY=0.85's -0.11m to -0.12m), not
tipping over. Of the sweep, DUTY=0.85/STRIDE=0.05/CYCLE_T=1.2/LIFT=0.03
had the best raw tilt (4.26deg peak) and speed (0.112 m/s) numbers in
isolation, but a follow-up full grasp-integration run with it showed a
real side effect: APPROACH does not steer on heading (see its own
docstring in quadruped_grasp_core.py), so net lateral drift during the
walk shows up directly as a worse final aim, and CYCLE_T=1.2's larger
-0.114m/30s drift (vs CYCLE_T=1.5's -0.056m/30s -- about half) produced
a visibly weaker final grasp force in that test. CYCLE_T=1.5 was kept
instead as the better trade for this specific task, at a small cost in
peak speed (0.103 vs 0.112 m/s) and peak tilt (5.38 vs 4.26deg, still
nowhere near concerning). Either way this is a ~3.4-3.8x speedup over
the old creep gait's ~0.03 m/s on this same (post-mass-fix) robot,
using two legs at once as asked. Re-validate with
sweep_trot.py/sweep_trot2.py (kept in this folder) before pushing
these numbers further -- they were picked from a modest six-point
sweep plus one grasp-integration comparison, not an exhaustive search.

Directionality bug and fix: with the stride swept symmetrically around
the stance-neutral foot position (STANCE_X +-STRIDE/2), the robot
consistently walked in -x -- AWAY from the head/trunk end (+x) it
needs to approach objects with -- and reversing the sign of the whole
stance/swing sweep did NOT reverse this (both directions still drifted
-x), ruling out a simple sign-convention mistake. Root cause, found via
leg_ik's own IK sweep data: the hip-angle-to-foot-x mapping is
NONLINEAR and NOT symmetric around the stance-neutral angle (e.g. a
+0.05m foot shift needs only ~6.6deg less hip angle than neutral, while
a -0.05m shift needs ~8.4deg more) -- a real consequence of the leg's
own changing lever arm at different angles, not a bug. That bakes an
asymmetric torque/tracking profile into the stride regardless of which
half is labelled "stance" vs "swing", biasing net motion toward -x no
matter the labelling. Fix: re-center the stride sweep at
(STANCE_X + X_OFFSET) instead of exactly STANCE_X, and calibrate
X_OFFSET empirically (offset_sweep.py) against ACTUAL measured net
displacement rather than assumed kinematics -- sweeping offsets found
a clear win/stable band around X_OFFSET=-0.045 (net +0.27m over 16s,
max tilt 3.37deg) versus a sharp instability cliff for any *positive*
offset beyond about +0.015 (full tip-over). -0.045 is what's used
below.
"""
import numpy as np
import leg_ik

STANCE_X = -2.11010576e-05     # foot x offset at neutral stance (~0)
STANCE_Z = -2.41800308e-01     # foot z offset at neutral stance
X_OFFSET = -0.045  # stride-center recalibration -- see module docstring

# 2026-09-23, Task #14: diagonal-trot values, validated via
# sweep_trot.py/sweep_trot2.py (see module docstring). Old creep-gait
# values were STRIDE=0.03/LIFT=0.02/DUTY=0.90/CYCLE_T=3.0.
STRIDE = 0.05     # peak-to-peak fore/aft foot travel, meters
LIFT = 0.03       # peak swing-phase foot rise above stance height, meters
DUTY = 0.85       # fraction of cycle each leg spends in stance
CYCLE_T = 1.5     # seconds per full gait cycle -- picked over the
                  # slightly-faster CYCLE_T=1.2 point from the sweep
                  # because the full grasp-integration test is sensitive
                  # to heading drift (APPROACH doesn't steer on azimuth,
                  # so any net lateral drift shows up directly as a worse
                  # final aim/weaker grasp): CYCLE_T=1.2 had -0.114m
                  # lateral drift over 30s vs 1.5's -0.056m -- about half
                  # -- for barely more speed (0.112 vs 0.103 m/s), so 1.5
                  # is the better trade for THIS task even though 1.2 was
                  # marginally faster and had marginally lower peak tilt.
SETTLE_T = 1.5    # seconds to hold pure stance before the gait starts,
                  # so the robot begins walking from the already-verified
                  # stable stand rather than mid-swing at t=0

# Diagonal-pair footfall: (fl, rr) swing together, then (fr, rl) --
# two legs moving at once, per direct user request. Supersedes the old
# one-quarter-cycle-apart single-leg order (see module docstring for
# why this is safe despite dropping to a support LINE during each
# pair's swing).
PHASE_OFFSET = {"fl": 0.00, "rr": 0.00, "fr": 0.50, "rl": 0.50}
LEGS = ("fl", "fr", "rl", "rr")


# Which side each leg is on, for differential-stride turning (see
# GaitController.joint_targets' turn_bias parameter below).
SIDE_SIGN = {"fl": +1.0, "rl": +1.0, "fr": -1.0, "rr": -1.0}


def foot_target(leg, t, turn_bias=0.0):
    """Return (x, z) target offset from this leg's stance neutral, at time t.

    turn_bias in [-1, 1] scales this leg's own stride length by
    (1 - turn_bias*SIDE_SIGN[leg]) -- i.e. it shortens one side's stride
    and lengthens the other's, tank-steer style, calibrated empirically
    the same way X_OFFSET was (see turn_calibration in the module
    docstring / walk_gait.py history)."""
    stride = STRIDE * (1.0 - turn_bias * SIDE_SIGN[leg])
    t_g = max(0.0, t - SETTLE_T)
    s = ((t_g / CYCLE_T) + PHASE_OFFSET[leg]) % 1.0
    if s < DUTY:
        u = s / DUTY
        x = stride / 2.0 - stride * u
        z = 0.0
    else:
        v = (s - DUTY) / (1.0 - DUTY)
        x = -stride / 2.0 + stride * v
        z = LIFT * np.sin(np.pi * v)
    return STANCE_X + X_OFFSET + x, STANCE_Z + z


def foot_target_manual(leg, t, direction=1.0, turn_bias=0.0):
    """Like foot_target, but for manual (interactive) walking in either
    direction -- Task #16 (manual_control_elephant_quadruped.py).

    direction=+1.0 is identical to foot_target (forward). direction=-1.0
    is NOT implemented as "run the same gait clock backward in time" --
    that was tried first and measured (test_walk_reverse.py-style check,
    2026-09-23) to be wrong: decrementing t does reverse the sign of net
    displacement, but the magnitude collapses to roughly 1/25th of forward
    speed over the same window (X_OFFSET below is calibrated specifically
    to correct a forward-direction asymmetry in the leg's own nonlinear
    hip-angle-to-foot-x mapping -- see the module docstring's `Directionality
    bug and fix` section -- and time-reversing the whole clock does not
    invert that correction the way it would need to). Instead this MIRRORS
    the stride sweep itself (t keeps advancing forward, normally) by
    negating `stride`, and -- measured, not assumed -- uses NO stride-center
    offset for backward (X_OFFSET is dropped, not flipped): a direct sweep
    found flipping the sign of X_OFFSET for backward walking produces a
    real net displacement but with periodic tilt spikes up to ~19deg and an
    inconsistent net direction over longer windows (16s), whereas dropping
    the offset entirely for backward gives a smaller but consistent and
    stable backward displacement (~-0.49m/16s, max tilt ~7.5deg, comparable
    to forward's own tilt levels) -- see test_walk_reverse3.py in the
    diagnostics kept in this folder. Net effect: backward walking works and
    is stable, but is noticeably slower than forward (~0.03 m/s vs ~0.10
    m/s) -- this was not tuned further beyond confirming it is safe and
    correctly-directioned, since manual backward is meant for small
    repositioning nudges, not sustained travel. Revisit offset_sweep.py's
    own methodology here if faster backward walking is ever needed."""
    stride = STRIDE * direction * (1.0 - turn_bias * SIDE_SIGN[leg])
    offset = X_OFFSET if direction >= 0.0 else 0.0
    t_g = max(0.0, t - SETTLE_T)
    s = ((t_g / CYCLE_T) + PHASE_OFFSET[leg]) % 1.0
    if s < DUTY:
        u = s / DUTY
        x = stride / 2.0 - stride * u
        z = 0.0
    else:
        v = (s - DUTY) / (1.0 - DUTY)
        x = -stride / 2.0 + stride * v
        z = LIFT * np.sin(np.pi * v)
    return STANCE_X + offset + x, STANCE_Z + z


class GaitController:
    """Holds per-leg IK warm-start state and returns joint targets."""

    def __init__(self):
        self._guess = {leg: (np.radians(21.57), np.radians(-43.13)) for leg in LEGS}

    def joint_targets(self, t, turn_bias=0.0):
        """Returns dict leg -> (abd_rad, hip_rad, knee_rad). Abduction is
        held at 0 -- see the module docstring for why an active lateral
        weight-shift via abduction was tried and dropped. turn_bias
        steers (see foot_target); 0.0 is straight ahead."""
        out = {}
        for leg in LEGS:
            x, z = foot_target(leg, t, turn_bias)
            h, k = leg_ik.solve_ik(x, z, guess=self._guess[leg])
            self._guess[leg] = (h, k)
            out[leg] = (0.0, h, k)
        return out

    def joint_targets_manual(self, t, direction=1.0, turn_bias=0.0):
        """Same as joint_targets, but using foot_target_manual so direction
        can be -1.0 for (calibrated, stable, slower) backward walking. See
        foot_target_manual's docstring for why this isn't just
        joint_targets(-t)."""
        out = {}
        for leg in LEGS:
            x, z = foot_target_manual(leg, t, direction, turn_bias)
            h, k = leg_ik.solve_ik(x, z, guess=self._guess[leg])
            self._guess[leg] = (h, k)
            out[leg] = (0.0, h, k)
        return out


if __name__ == "__main__":
    # Sanity check: print one full cycle's targets for a single leg
    # (after the settle window) and confirm continuity at the
    # stance/swing boundary (no jump in x or z).
    for frac in np.linspace(0, 1, 21):
        t = SETTLE_T + frac * CYCLE_T
        x, z = foot_target("fl", t)
        print(f"t={t:5.3f}s frac={frac:.2f} x={x:+.4f} z={z:+.4f}")
