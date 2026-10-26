import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY: one headless stack run starting with the box at the park spot (+noise offsets);
prints one JSON line with ground-truth scoring. Parameters via env (WRAP_*, PARK_X, ...)."""
import sys, os, json, math, numpy as np, mujoco
import quadruped_grasp_core as core, elephant_grasp_cv as cv, ground_truth_eval as gt
m, d = core.model, core.data
def run(dx=0.0, dy=0.0, yaw_deg=0.0, max_t=70.0):
    mujoco.mj_resetDataKeyframe(m, d, 0); core.reset_commands(); gt.settle(600)
    j = m.joint('torso_free').qposadr[0]; h = math.radians(yaw_deg)
    d.qpos[j+3:j+7] = [math.cos(h/2), 0, 0, math.sin(h/2)]; gt.settle(300)
    # box at the nominal approach spot in world (robot at origin facing +x), slightly further than park
    d.qpos[gt.CUBE_J:gt.CUBE_J+7] = [cv.PARK_X + 0.45 + dx, cv.PARK_Y * 1.0 + dy, 0.1005, 1, 0, 0, 0]
    mujoco.mj_forward(m, d); gt.settle(200)
    st = cv.AutonomousStack(verbose=False); st.start(); dt = m.opt.timestep
    best_up_lift = 0.0; maxup_t = 0.0; min_up = 1.0; phases = {}
    last = None
    while st.active and st.t < max_t:
        d.ctrl[:] = 0; st.step(dt); mujoco.mj_step(m, d)
        if st.state == 'wrap':
            up = float(d.xmat[gt.CUBE].reshape(3, 3)[2, 2]); z = 100*(d.xpos[gt.CUBE][2]-0.1005)
            min_up = min(min_up, up)
            if up > 0.6: best_up_lift = max(best_up_lift, z)
            ph = st.wrap.phase
            if ph != last:
                tr = gt.truth_in_torso(); phases[ph] = [round(float(tr[0]),3), round(float(tr[1]),3), round(up,2)]; last = ph
    up = float(d.xmat[gt.CUBE].reshape(3, 3)[2, 2]); z = 100*(d.xpos[gt.CUBE][2]-0.1005)
    return dict(state=st.state, final_lift=round(float(z),2), up=round(up,2), best_up_lift=round(float(best_up_lift),2),
                min_up=round(min_up,2), t=round(st.t,1), phases=phases)
if __name__ == "__main__":
    a = [float(x) for x in sys.argv[1:]] or [float(os.environ.get("TR_DX", 0)), float(os.environ.get("TR_DY", 0)), float(os.environ.get("TR_YAW", 0))]
    print("RESULT " + json.dumps(run(*a)), flush=True)
