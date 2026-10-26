import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY. Compares the camera-only box estimates against the simulator's
true box pose. This file (and only this file) reads ground truth: the control
code never imports it. Usage: python3 ground_truth_eval.py"""
import math, sys, time
import numpy as np
import mujoco
import quadruped_grasp_core as core
import vision

model, data = core.model, core.data
CUBE = model.body("test_cube").id
CUBE_J = model.joint("cube_free").qposadr[0]


def settle(n=600):
    for _ in range(n):
        data.ctrl[:] = 0
        core.apply_legs(core.STAND_TARGETS)
        core.apply_neck(0.0)
        mujoco.mj_step(model, data)


def truth_in_torso():
    R = data.xmat[core.torso_body].reshape(3, 3)
    p = R.T @ (data.xpos[CUBE] - data.xpos[core.torso_body])
    return p


def place_box(x_fwd, y_left):
    """Place box at (x_fwd, y_left) in the torso frame, standing on the floor."""
    R = data.xmat[core.torso_body].reshape(3, 3)
    w = data.xpos[core.torso_body] + R @ np.array([x_fwd, y_left, 0.0])
    data.qpos[CUBE_J:CUBE_J + 7] = [w[0], w[1], 0.1005, 1, 0, 0, 0]
    data.qvel[model.joint("cube_free").dofadr[0]:model.joint("cube_free").dofadr[0] + 6] = 0
    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    settle(800)
    print("base_height (kinematic) = %.3f ; true torso z = %.3f" % (vision.base_height(), data.xpos[core.torso_body][2]))
    rows = []
    for x in (1.5,2.0,2.5,3.0,4.0,5.0):
        for y in (-0.5, 0.0, 0.5):
            place_box(x, y); settle(150)
            t = truth_in_torso()
            blobs, est = vision.see(vision.FAR_CAMS)
            per = {}
            for c, b in blobs.items():
                e = vision.estimate_from_blob(b)
                if e: per[c] = (e.x - t[0], e.y - t[1], e.kind)
            fused = (est.x - t[0], est.y - t[1]) if est else None
            print(f"box@({t[0]:.2f},{t[1]:+.2f})  seen={list(blobs)}  " +
                  "  ".join(f"{c}: dx={v[0]:+.3f} dy={v[1]:+.3f} [{v[2]}]" for c, v in per.items()) +
                  (f"   FUSED dx={fused[0]:+.3f} dy={fused[1]:+.3f}" if fused else "   FUSED none"))
