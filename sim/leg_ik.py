"""
Numeric (measured, not hand-derived) 2-link leg IK for the elephant
quadruped's hip/knee pair, used by the gait generator (walk_gait.py).

Why numeric instead of closed-form: the closed-form symmetric relation
knee = -2*hip (used to size the standing pose in build_elephant_quadruped.py)
only ever produces zero horizontal foot travel -- by construction it is a
pure "squat" locus, not a stride locus. Any real stride needs to break
that symmetry, and rather than re-derive the general 2-link analytic IK
by hand (algebra-sign-error-prone, as this project has been burned by
before), this solves it the same way this project settled the standing
kp value: by direct measurement in the actual simulator (mj_kinematics),
via Newton iteration on a numerically-measured Jacobian. All four legs
share the same thigh/shank geometry and the same hip/knee axis
(both "0 1 0", pitch-only in the sagittal x-z plane), so one solver
serves all four; only the abduction joint and the hip's own (x,y)
mount position differ per leg, and abduction is left at 0 for this
first-phase straight-line gait.
"""
import os

import mujoco
import numpy as np

# 2026-09-23, later still again (Task #16 follow-up): same cwd-dependent
# relative-path bug as quadruped_grasp_core.py's MODEL_FILE (see its
# comment) -- this module loaded its own separate MjModel instance for IK
# math, using the same bare filename, so it was equally vulnerable to
# silently loading a stale XML from whatever directory the process happened
# to be launched from. Anchored to this file's own directory for the same
# reason.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "robot_elephant_quadruped.xml")

_MODEL = None
_DATA = None
_HIP_QADR = None
_KNEE_QADR = None
_HIP_BODY = None
_FOOT_SITE = None


def _ensure_loaded():
    global _MODEL, _DATA, _HIP_QADR, _KNEE_QADR, _HIP_BODY, _FOOT_SITE
    if _MODEL is not None:
        return
    _MODEL = mujoco.MjModel.from_xml_path(_MODEL_PATH)
    _DATA = mujoco.MjData(_MODEL)
    hip_jid = _MODEL.joint("j_fl_hip").id
    knee_jid = _MODEL.joint("j_fl_knee").id
    _HIP_QADR = _MODEL.jnt_qposadr[hip_jid]
    _KNEE_QADR = _MODEL.jnt_qposadr[knee_jid]
    _HIP_BODY = _MODEL.body("hip_fl").id
    _FOOT_SITE = _MODEL.site("foot_site_fl").id


def foot_rel_xz(hip_rad, knee_rad):
    """Foot position relative to the hip body, in the sagittal (x,z) plane."""
    _ensure_loaded()
    _DATA.qpos[_HIP_QADR] = hip_rad
    _DATA.qpos[_KNEE_QADR] = knee_rad
    mujoco.mj_kinematics(_MODEL, _DATA)
    rel = _DATA.site_xpos[_FOOT_SITE] - _DATA.xpos[_HIP_BODY]
    return np.array([rel[0], rel[2]])


def solve_ik(x_target, z_target, guess=(0.377, -0.753), iters=25, tol=1e-5):
    """Newton-solve (hip, knee) so foot_rel_xz(hip, knee) == (x_target, z_target).

    guess defaults to the validated standing-stance angles (radians).
    Returns (hip_rad, knee_rad); raises if it fails to converge (caller
    should keep targets inside the mechanically reachable band).
    """
    q = np.array(guess, dtype=float)
    target = np.array([x_target, z_target])
    h = 1e-4
    for _ in range(iters):
        f0 = foot_rel_xz(q[0], q[1])
        err = target - f0
        if np.linalg.norm(err) < tol:
            return q[0], q[1]
        f_h = foot_rel_xz(q[0] + h, q[1])
        f_k = foot_rel_xz(q[0], q[1] + h)
        J = np.column_stack([(f_h - f0) / h, (f_k - f0) / h])
        try:
            dq = np.linalg.solve(J, err)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(J, err, rcond=None)[0]
        # damp the step to avoid overshoot into a singular/unreachable region
        step = np.clip(dq, -0.5, 0.5)
        q = q + step
    raise RuntimeError(
        f"IK did not converge for target=({x_target:.4f},{z_target:.4f}): "
        f"final err={np.linalg.norm(err):.5f}, q={q}"
    )


if __name__ == "__main__":
    # Self-check: recover the known standing-stance angles from their own
    # foot position, and confirm a small stride offset solves cleanly.
    stance = foot_rel_xz(np.radians(21.57), np.radians(-43.13))
    print("stance foot_rel:", stance)
    h, k = solve_ik(stance[0], stance[1])
    print(f"round-trip solve -> hip={np.degrees(h):.3f}deg knee={np.degrees(k):.3f}deg "
          f"(expected 21.57 / -43.13)")

    for dx in (-0.05, -0.02, 0.0, 0.02, 0.05):
        x = stance[0] + dx
        z = stance[1]
        h, k = solve_ik(x, z, guess=(h, k))
        check = foot_rel_xz(h, k)
        print(f"x={x:+.4f} z={z:+.4f} -> hip={np.degrees(h):+7.3f}deg "
              f"knee={np.degrees(k):+7.3f}deg  check={check}")

    print("\nlift test at stance x, raised z:")
    for dz in (0.0, 0.02, 0.04, 0.06):
        x = stance[0]
        z = stance[1] + dz
        h, k = solve_ik(x, z, guess=(np.radians(21.57), np.radians(-43.13)))
        check = foot_rel_xz(h, k)
        print(f"x={x:+.4f} z={z:+.4f} -> hip={np.degrees(h):+7.3f}deg "
              f"knee={np.degrees(k):+7.3f}deg  check={check}")
