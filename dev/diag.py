import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY: run the stack headless and print the TRUE box pose (torso frame), tilt, lift, and
trunk-box contact at every wrap phase change."""
import sys, math, os, numpy as np, mujoco
import quadruped_grasp_core as core, elephant_grasp_cv as cv, ground_truth_eval as gt
m, d = core.model, core.data
TRUNK = {b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or '').startswith('t1_u')}
def trunk_contacts():
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 == gt.CUBE and b2 in TRUNK) or (b2 == gt.CUBE and b1 in TRUNK): n += 1
    return n
def snap(tag):
    tr = gt.truth_in_torso(); up = float(d.xmat[gt.CUBE].reshape(3, 3)[2, 2])
    print(f"  [{tag:8s}] box_true=({tr[0]:.3f},{tr[1]:+.3f}) up={up:.2f} z={100*(d.xpos[gt.CUBE][2]-0.1005):+.1f}cm trunk_contacts={trunk_contacts()}", flush=True)
def run(bx=1.05, by=0.0, yaw_deg=0.0, max_t=90.0):
    mujoco.mj_resetDataKeyframe(m, d, 0); core.reset_commands(); gt.settle(600)
    j = m.joint('torso_free').qposadr[0]; h = math.radians(yaw_deg)
    d.qpos[j+3:j+7] = [math.cos(h/2), 0, 0, math.sin(h/2)]; gt.settle(300)
    d.qpos[gt.CUBE_J:gt.CUBE_J+7] = [bx, by, 0.1005, 1, 0, 0, 0]; mujoco.mj_forward(m, d); gt.settle(200)
    st = cv.AutonomousStack(verbose=False); st.start(); dt = m.opt.timestep
    last = None; maxcon = 0; mintilt = 1.0
    while st.active and st.t < max_t:
        d.ctrl[:] = 0; st.step(dt); mujoco.mj_step(m, d)
        if st.state == 'wrap':
            maxcon = max(maxcon, trunk_contacts()); mintilt = min(mintilt, float(d.xmat[gt.CUBE].reshape(3, 3)[2, 2]))
            ph = st.wrap.phase
            if ph != last:
                if last is not None: snap("end " + last)
                last = ph
    snap("final")
    up = float(d.xmat[gt.CUBE].reshape(3, 3)[2, 2])
    res = dict(state=st.state, lift_cm=round(100*(d.xpos[gt.CUBE][2]-0.1005), 1), up=round(up, 2), min_up=round(mintilt, 2), max_contacts=maxcon)
    print(res, flush=True); return res
if __name__ == "__main__":
    a = [float(x) for x in sys.argv[1:]]; run(*a)
