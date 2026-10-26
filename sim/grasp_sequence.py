"""The wrap-and-lift choreography (neck/trunk curl, cable squeeze/cinch,
lift, crouch release), unchanged from the earlier autopilot except for one
thing: the cable-pull azimuth psi is now PASSED IN (from the camera-derived
box position, see vision.psi_from_estimate) instead of being computed from
the simulator's true box pose.

Open loop by design: it plays a fixed, smooth, cosine-eased schedule. It
needs the robot already parked at the right distance from the box --
choosing that distance is the autonomy stack's job (elephant_grasp_cv.py).
"""
import math

import numpy as np

import quadruped_grasp_core as core
import walk_gait
import leg_ik

model, data = core.model, core.data
TP_ACT, TD_ACT = core.TP_ACT, core.TD_ACT

CABLE_DIRS = ((1.0, 0.0), (-0.5, math.sqrt(3) / 2), (-0.5, -math.sqrt(3) / 2))


def cable_vec(psi, M):
    """Exact-decomposition cable-tension solve: pull of magnitude M toward azimuth psi."""
    if M <= 0:
        return [0.0, 0.0, 0.0]
    p = np.array([math.cos(math.radians(psi)), math.sin(math.radians(psi))])
    for a, b in ((0, 1), (1, 2), (0, 2)):
        co = np.linalg.solve(np.array([CABLE_DIRS[a], CABLE_DIRS[b]]).T, p)
        if co.min() >= -1e-9:
            w = [0.0, 0.0, 0.0]
            w[a], w[b] = max(0, co[0]), max(0, co[1])
            s = max(w)
            return [M * x / s for x in w]


# ---- WRAP SETTINGS (hand-tunable) ---------------------------------------------------------
# Flow:  the robot stops with the box ~0.70 m out and ~20 deg off its nose (see PARK_* in
# elephant_grasp_cv.py)  ->  TURN: yaw the robot (camera-guided) with the neck/trunk/knees NOT yet bent,
# until the box is at the right sideways offset  ->  REACH: only now bend the neck, trunk and knees to
# lower the trunk onto the box's side, no cable pull
# ->  (optional, only if WRAP_ADV>0) ADVANCE: creep the ROBOT forward (trunk passive, no cable pull) until the CAMERA says the
# trunk is touching the box  ->  SQUEEZE/CINCH: only now pull the cables so the trunk curls
# round the box  ->  lift (cable pull, then legs straighten).
# The numbers below are the simulation-tuned values that lifted the box ~4 cm when the box was
# already at (0.50 m, 0.093 m left) -- ADVANCE just brings the box to that same relative spot.
# NOTE: those were tuned for the ORIGINAL stiff trunk and 60 N cables. The defaults for m1/m2 are
# scaled by the trunk-softness factor (TRUNK_STIFF) and m3 is capped at CABLE_MAX_N, but that is a
# first guess -- expect to re-tune m1/m2/m3 by hand.
# Every value can be overridden without editing code:  WRAP_NK=40 WRAP_ADV=0.15 mjpython ...
#   nk   neck bend (deg)           tp   trunk bend (deg, negative = down)   td  trunk extension (m, 0..0.12)
#   cr   crouch depth (m)          psi  cable curl direction (deg; 0 = ahead/down, +90 = sideways toward the box;
#                                        used as-is for a box on the left, mirrored for the right)
#   m1/m2/m3  cable pull (N) while squeezing / cinching / lifting (max 60)
#   t_reach seconds to lower the trunk; t_up / t_legs seconds for the lift phases
#   adv  how far to roll forward before curling (m);  adv_dir  wheel speed for that roll (0..1)
_DEFAULTS = dict(nk=24.21, tp=-71.14, td=0.1095, cr=0.0886, t_reach=4.0,
                 m1=10.20, m2=19.56, m3=min(29.95, core.CABLE_MAX_N), psi=42.96, t_up=4.99, t_legs=4.61,
                 adv=0.0, adv_dir=0.30)    # CMA-ES-tuned in simulation for the thin box, soft trunk, 30 N cables
import os as _os
TUNED = {k: float(_os.environ.get("WRAP_" + k.upper(), v)) for k, v in _DEFAULTS.items()}
PSI_FALLBACK = TUNED["psi"]
M_SQUEEZE, M_CINCH, M_LIFT = TUNED["m1"], TUNED["m2"], TUNED["m3"]
NK_DEG, TP_DEG, TD_M, CROUCH_M = TUNED["nk"], TUNED["tp"], TUNED["td"], TUNED["cr"]
ADV_DIR = max(0.05, min(1.0, TUNED["adv_dir"]))
ADV_T = max(0.0, TUNED["adv"]) / (0.258 * ADV_DIR)    # 0.258 m/s per unit wheel command, measured in sim
STEP_PHASES = set()

SEQUENCE = [
    ("TURN",    3.0, None, None, None, "NONE", None),      # neck/trunk/legs stay as they were; only the wheels move
    ("REACH",   TUNED["t_reach"], NK_DEG, TP_DEG, TD_M, "NONE", CROUCH_M),
    ("ADVANCE", max(ADV_T, 0.01), NK_DEG, TP_DEG, TD_M, "NONE", CROUCH_M),
    ("SQUEEZE", 1.5, NK_DEG, TP_DEG, TD_M, "SQ", CROUCH_M),
    ("SETTLE1", 1.0, NK_DEG, TP_DEG, TD_M, "SQ", CROUCH_M),
    ("CINCH",   2.5, NK_DEG, TP_DEG, TD_M, "CI", CROUCH_M),
    ("HOLD",    0.8, NK_DEG, TP_DEG, TD_M, "CI", CROUCH_M),
    ("DROPUP",  TUNED["t_up"], NK_DEG, TP_DEG, 0.0,  "LI", CROUCH_M),
    ("HOLD2",   1.0, NK_DEG, TP_DEG, 0.0,  "LI", CROUCH_M),
    ("LEGS",    TUNED["t_legs"], NK_DEG, TP_DEG, 0.0,  "LI", 0.0),
    ("CARRY",   2.0, NK_DEG, TP_DEG, 0.0,  "LI", 0.0),
]

if TUNED["adv"] <= 0:                      # no forward roll: the robot only turns, then curls
    SEQUENCE = [ph for ph in SEQUENCE if ph[0] != "ADVANCE"]

HK_GUESS = (math.radians(21.57), math.radians(-43.13))


class BoxWrap:
    """Live, per-tick version of the staged wrap sequence."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.state = "idle"   # idle -> running -> holding
        self.leg_guess = {l: HK_GUESS for l in walk_gait.LEGS}
        self.phase_i = 0
        self.phase_t = 0.0
        self.prev = (0.0, 0.0, 0.0, np.zeros(3), 0.0)
        self.hold_phases = set()    # phases the stack keeps open until its cameras say they are done (ADVANCE, TURN)
        self._cable_by_mode = {"NONE": [0.0] * 3, "SQ": [0.0] * 3, "CI": [0.0] * 3, "LI": [0.0] * 3}

    @property
    def phase(self):
        return SEQUENCE[min(self.phase_i, len(SEQUENCE) - 1)][0]

    def start(self, psi_deg=None):
        psi = PSI_FALLBACK if psi_deg is None else psi_deg
        nk0 = math.degrees(float(data.ctrl[core.NECK_ACT]))
        tp0 = math.degrees(float(data.ctrl[TP_ACT]))
        td0 = float(data.ctrl[TD_ACT])
        c0 = np.array([float(data.ctrl[core.T[i]]) for i in range(3)])
        self.prev = (nk0, tp0, td0, c0, 0.0)
        self.phase_i = 0
        self.phase_t = 0.0
        self.state = "running"
        self._cable_by_mode = {"NONE": [0.0] * 3, "SQ": cable_vec(psi, M_SQUEEZE),
                               "CI": cable_vec(psi, M_CINCH), "LI": cable_vec(psi, M_LIFT)}
        if self.verbose:
            print(f"[BOXWRAP] start -- phase {SEQUENCE[0][0]}, cable azimuth psi={psi:.1f} deg "
                  f"({'camera' if psi_deg is not None else 'FALLBACK, no camera estimate'})")

    def release_phase(self, name):
        """End a held phase (ADVANCE = roll forward, TURN = yaw toward the box) now."""
        self.hold_phases.discard(name)
        if self.state == "running" and self.phase == name:
            self.phase_t = SEQUENCE[self.phase_i][1]

    def abort(self):
        if self.state != "idle" and self.verbose:
            print("[BOXWRAP] aborted")
        self.state = "idle"

    def step(self, dt):
        if self.state == "idle":
            return
        idx = min(self.phase_i, len(SEQUENCE) - 1)
        name, dur, nk, tp, td, cb_mode, cr = SEQUENCE[idx]
        cb = self._cable_by_mode[cb_mode]
        if self.state == "running":
            if name not in self.hold_phases:
                self.phase_t += dt
            f = 1.0 if name in STEP_PHASES else min(1.0, self.phase_t / dur)
        else:
            f = 1.0
        f = 0.5 - 0.5 * math.cos(math.pi * f)
        p_nk, p_tp, p_td, p_cb, p_cr = self.prev
        nk = p_nk if nk is None else nk          # None = keep the previous pose (TURN: don't bend yet)
        tp = p_tp if tp is None else tp
        td = p_td if td is None else td
        cr = p_cr if cr is None else cr
        cur_nk = p_nk + (nk - p_nk) * f
        cur_tp = p_tp + (tp - p_tp) * f
        cur_td = p_td + (td - p_td) * f
        cur_cb = p_cb + (np.array(cb) - p_cb) * f
        cur_cr = p_cr + (cr - p_cr) * f

        core.apply_neck(math.radians(cur_nk))
        data.ctrl[TP_ACT] = float(np.clip(math.radians(cur_tp), *model.actuator_ctrlrange[TP_ACT]))
        data.ctrl[TD_ACT] = float(np.clip(cur_td, *model.actuator_ctrlrange[TD_ACT]))
        for i in range(3):
            data.ctrl[core.T[i]] = cur_cb[i]
            core.current[i] = cur_cb[i]

        legs = {}
        for l in walk_gait.LEGS:
            h, k = leg_ik.solve_ik(walk_gait.STANCE_X + walk_gait.X_OFFSET,
                                   walk_gait.STANCE_Z + cur_cr, guess=self.leg_guess[l])
            self.leg_guess[l] = (h, k)
            legs[l] = (0.0, h, k)
        core.apply_legs(legs)

        if self.state == "running" and self.phase_t >= dur:
            self.prev = (nk, tp, td, np.array(cb, float), cr)
            self.phase_i += 1
            self.phase_t = 0.0
            if self.phase_i >= len(SEQUENCE):
                self.state = "holding"
                if self.verbose:
                    print("[BOXWRAP] sequence complete -- holding CARRY pose.")
            elif self.verbose:
                print(f"[BOXWRAP] phase: {SEQUENCE[self.phase_i][0]}")
