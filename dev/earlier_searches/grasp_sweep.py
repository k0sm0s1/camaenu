import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY: find the parking distance from which the open-loop wrap actually
lifts the box. Uses ground truth only to MEASURE the result (box height)."""
import sys, math, time
import numpy as np, mujoco
import quadruped_grasp_core as core, vision, grasp_sequence as gs
import ground_truth_eval as gt
model, data = core.model, core.data

def run(x, y, use_vision_psi=True, verbose=False, psi_override=None):
    mujoco.mj_resetDataKeyframe(model, data, 0); core.reset_commands()
    gt.settle(800); gt.place_box(x, y); gt.settle(200)
    blobs, est = vision.see()
    psi = psi_override if psi_override is not None else (vision.psi_from_estimate(est) if use_vision_psi else None)
    t_true = gt.truth_in_torso()
    wrap = gs.BoxWrap(verbose=verbose); wrap.start(psi)
    dt = model.opt.timestep
    zmax = 0; log = []
    n = 0
    while wrap.state == "running":
        data.ctrl[:] = 0; wrap.step(dt); mujoco.mj_step(model, data); n += 1
        if n % 100 == 0:
            z = data.xpos[gt.CUBE][2]; zmax=max(zmax,z); log.append((wrap.phase, round(z, 3)))
    zc = data.xpos[gt.CUBE][2]
    up = data.xmat[gt.CUBE].reshape(3,3)[2,2]
    return dict(x=x, y=y, psi=None if psi is None else round(psi,1), est=None if est is None else (round(est.x,3), round(est.y,3)),
                true=(round(t_true[0],3), round(t_true[1],3)), lift_cm=round((zc-0.1005)*100,1), maxlift_cm=round((zmax-0.1005)*100,1), upright=round(float(up),2), log=None if not verbose else log)

if __name__ == "__main__":
    x=float(sys.argv[1]); y=float(sys.argv[2]); psi=None if sys.argv[3]=="v" else float(sys.argv[3])
    r = run(x, y, psi_override=psi)
    print(f"x={r['x']} y={r['y']} psi={r['psi']} est={r['est']} lift_cm={float(r['lift_cm'])} maxlift_cm={float(r['maxlift_cm'])} upright={r['upright']}", flush=True)
