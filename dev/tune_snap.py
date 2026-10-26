import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY: run the camera stack headless until the TURN phase ends (wrap enters REACH), then save the
full physics state. Wrap-only tuning (feas3.py) restores these states, so it sees the real pipeline's
park/turn variability without paying for rendering."""
import sys, os, math, numpy as np, mujoco
import quadruped_grasp_core as core, elephant_grasp_cv as cv, ground_truth_eval as gt
m, d = core.model, core.data
tag = sys.argv[1]; dx, dy, yaw_deg = [float(x) for x in sys.argv[2:5]]
mujoco.mj_resetDataKeyframe(m, d, 0); core.reset_commands(); gt.settle(600)
j = m.joint('torso_free').qposadr[0]; h = math.radians(yaw_deg)
d.qpos[j+3:j+7] = [math.cos(h/2), 0, 0, math.sin(h/2)]; gt.settle(300)
d.qpos[gt.CUBE_J:gt.CUBE_J+7] = [cv.PARK_X + 0.45 + dx, cv.PARK_Y + dy, 0.1005, 1, 0, 0, 0]
mujoco.mj_forward(m, d); gt.settle(200)
st = cv.AutonomousStack(verbose=False); st.start(); dt = m.opt.timestep
while st.active and st.t < 60:
    d.ctrl[:] = 0; st.step(dt); mujoco.mj_step(m, d)
    if st.state == 'wrap' and st.wrap.phase == 'REACH':
        tr = gt.truth_in_torso()
        np.savez(f"/tmp/snaps/{tag}.npz", qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl, act=d.act, time=d.time,
                 warm=d.qacc_warmstart, cur=np.array(core.current), box=tr[:2])
        print(f"SNAP {tag} box_true=({tr[0]:.3f},{tr[1]:+.3f})", flush=True); sys.exit(0)
print("NO SNAP", st.state, st.reason)
