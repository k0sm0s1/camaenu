import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY: wrap-only run (no cameras). Box placed at (x,y) in the torso frame at the START of REACH
(i.e. just after TURN). Scores with ground truth."""
import sys, math, time, json
import numpy as np, mujoco
import quadruped_grasp_core as core, grasp_sequence as gs
import ground_truth_eval as gt
model, data = core.model, core.data
CUBE0 = 0.1005
SETTLE_EXTRA = int(__import__("os").environ.get("SETTLE_EXTRA", 0))
DEF = dict(nk=30.93, tp=-76.92, td=0.0956, cr=0.0704, t_reach=4.0, m1=10.59, m2=17.502, m3=30.0, psi=59.94, t_up=5.57, t_legs=4.09)
def build(p):
    nk, tp, td, cr = p["nk"], p["tp"], p["td"], p["cr"]
    return [("REACH", p["t_reach"], nk, tp, td, "NONE", cr), ("SQUEEZE", 1.5, nk, tp, td, "SQ", cr),
            ("SETTLE1", 1.0, nk, tp, td, "SQ", cr), ("CINCH", 2.5, nk, tp, td, "CI", cr), ("HOLD", 0.8, nk, tp, td, "CI", cr),
            ("DROPUP", p["t_up"], nk, tp, 0.0, "LI", cr), ("HOLD2", 1.0, nk, tp, 0.0, "LI", cr),
            ("LEGS", p["t_legs"], nk, tp, 0.0, "LI", 0.0), ("CARRY", 2.0, nk, tp, 0.0, "LI", 0.0)]
def run(p, x, y):
    q = dict(DEF); q.update(p)
    mujoco.mj_resetDataKeyframe(model, data, 0); core.reset_commands(); gt.settle(800); gt.settle(SETTLE_EXTRA); gt.place_box(x, y); gt.settle(200)
    gs.SEQUENCE[:] = build(q); gs.STEP_PHASES.clear()
    gs.M_SQUEEZE, gs.M_CINCH, gs.M_LIFT = q["m1"], q["m2"], q["m3"]
    wrap = gs.BoxWrap(verbose=False); wrap.start(q["psi"])
    dt = model.opt.timestep; n = 0; zmax = 0.0
    while wrap.state == "running":
        data.ctrl[:] = 0; wrap.step(dt); mujoco.mj_step(model, data); n += 1
        if n % 50 == 0:
            z = data.xpos[gt.CUBE][2]; up = data.xmat[gt.CUBE].reshape(3, 3)[2, 2]
            if up > 0.8: zmax = max(zmax, z)
            if not np.isfinite(z): break
    z = data.xpos[gt.CUBE][2]; up = data.xmat[gt.CUBE].reshape(3, 3)[2, 2]
    return dict(final=round(float((z - CUBE0) * 100), 2), best=round(float((zmax - CUBE0) * 100), 2) if zmax else -99.0, up=round(float(up), 2))
if __name__ == "__main__":
    t = time.time()
    for (x, y) in [(0.40, 0.073)]:
        print((x, y), run({}, x, y), round(time.time() - t), flush=True)
