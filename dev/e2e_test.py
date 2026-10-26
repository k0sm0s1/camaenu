import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY end-to-end run of the camera-only stack, headless. Ground truth is
read here (and only here) to place the box and to score the result."""
import sys, math, time
import numpy as np, mujoco
import quadruped_grasp_core as core, vision
import elephant_grasp_cv as cv
import ground_truth_eval as gt
model, data = core.model, core.data

def yaw():
    R = data.xmat[core.torso_body].reshape(3,3); return math.atan2(R[1,0], R[0,0])

def run(bx, by, robot_yaw_deg=0.0, max_t=90.0, verbose=True):
    mujoco.mj_resetDataKeyframe(model, data, 0); core.reset_commands()
    gt.settle(600)
    # put the BOX at world (bx,by) and rotate the ROBOT to face away from it by robot_yaw
    j=model.joint('torso_free').qposadr[0]
    h=math.radians(robot_yaw_deg); data.qpos[j+3:j+7]=[math.cos(h/2),0,0,math.sin(h/2)]
    gt.settle(300)
    data.qpos[gt.CUBE_J:gt.CUBE_J+7]=[bx,by,0.1005,1,0,0,0]; mujoco.mj_forward(model,data); gt.settle(200)
    st = cv.AutonomousStack(verbose=verbose); assert st.start()
    dt = model.opt.timestep; n=0; t0=time.time(); wrap_start=None
    while st.active and st.t < max_t:
        data.ctrl[:] = 0
        st.step(dt); mujoco.mj_step(model, data); n+=1
        if st.state=='wrap' and wrap_start is None:
            wrap_start = gt.truth_in_torso().copy()
    tr = gt.truth_in_torso()
    up = float(data.xmat[gt.CUBE].reshape(3,3)[2,2])
    out = dict(state=st.state, reason=st.reason, sim_t=round(st.t,1), wall=round(time.time()-t0),
               wrap_start_true=None if wrap_start is None else tuple(np.round(wrap_start[:2],3)),
               park_err=None if wrap_start is None else tuple(np.round(wrap_start[:2]-np.array([cv.PARK_X,cv.PARK_Y]),3)),
               box_end_true=tuple(np.round(tr[:2],3)), lift_cm=round((data.xpos[gt.CUBE][2]-0.1005)*100,1), upright=round(up,2),
               visual_lift_cm=None if st.visual_lift_m is None else round(st.visual_lift_m*100,1))
    return out

if __name__ == "__main__":
    bx,by,ry = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv)>3 else 0.0
    print(run(bx,by,ry))
