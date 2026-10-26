import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY feasibility search: can ANY open-loop wrap parameter set lift the box?
Uses ground truth (true box pose) for placement + scoring only."""
import sys, math, time, json
import numpy as np, mujoco
import quadruped_grasp_core as core, grasp_sequence as gs
import ground_truth_eval as gt
model, data = core.model, core.data
CUBE0 = 0.1005

def build(p):
    nk, tp, td, cr = p["nk"], p["tp"], p["td"], p["cr"]
    return [("REACH", p["t_reach"], nk, tp, td, "NONE", cr),
            ("SQUEEZE", 1.5, nk, tp, td, "SQ", cr),
            ("SETTLE1", 1.0, nk, tp, td, "SQ", cr),
            ("CINCH", 2.5, nk, tp, td, "CI", cr),
            ("HOLD", 0.8, nk, tp, td, "CI", cr),
            ("DROPUP", p.get("t_up",6.0), nk, tp, 0.0, "LI", cr),
            ("HOLD2", 1.0, nk, tp, 0.0, "LI", cr),
            ("LEGS", p.get("t_legs",5.0), nk, tp, 0.0, "LI", 0.0),
            ("CARRY", 2.0, nk, tp, 0.0, "LI", 0.0)]

def run(p, x=0.6, y=0.0):
    mujoco.mj_resetDataKeyframe(model, data, 0); core.reset_commands()
    gt.settle(800); gt.place_box(p.get("x", x), y + p.get("y", 0.0)); gt.settle(200)
    gs.SEQUENCE[:] = build(p)
    gs.STEP_PHASES.clear()                      # smooth REACH
    gs.M_SQUEEZE, gs.M_CINCH, gs.M_LIFT = p["m1"], p["m2"], p["m3"]
    wrap = gs.BoxWrap(verbose=False); wrap.start(p["psi"])
    dt = model.opt.timestep; n = 0; zmax = 0; best_hold = 0
    while wrap.state == "running":
        data.ctrl[:] = 0; wrap.step(dt); mujoco.mj_step(model, data); n += 1
        if n % 50 == 0:
            z = data.xpos[gt.CUBE][2]
            up = data.xmat[gt.CUBE].reshape(3, 3)[2, 2]
            if up > 0.9: zmax = max(zmax, z)
            if not np.isfinite(z): break
    z = data.xpos[gt.CUBE][2]; up = data.xmat[gt.CUBE].reshape(3, 3)[2, 2]
    return dict(final_lift_cm=(z - CUBE0) * 100, max_lift_cm=(zmax - CUBE0) * 100 if zmax else -99, upright=float(up))

BASE = dict(nk=55, tp=-50, td=0.12, cr=0.07, t_reach=4.0, m1=15, m2=50, m3=60, psi=-174.0, x=0.60, y=0.0)
if __name__ == "__main__":
    t = time.time(); r = run(BASE); print(r, "wall", round(time.time() - t, 1))
