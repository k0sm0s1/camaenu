import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY: place the tip beside the box at COM height, curl, measure wrap coverage."""
import sys, math, itertools, numpy as np, mujoco
from multiprocessing import Pool
def one(a):
    x,y,psi,pose=a
    import quadruped_grasp_core as core, grasp_sequence as gs, ground_truth_eval as gt, feas
    model,data=core.model,core.data
    nk,tp,td,cr=pose
    p=dict(nk=nk,tp=tp,td=td,cr=cr,t_reach=4.0,m1=15,m2=40,m3=60,psi=psi)
    mujoco.mj_resetDataKeyframe(model,data,0); core.reset_commands(); gt.settle(800); gt.place_box(x,y); gt.settle(200)
    gs.SEQUENCE[:]=feas.build(p)[:5]; gs.STEP_PHASES.clear()
    gs.M_SQUEEZE,gs.M_CINCH,gs.M_LIFT=p["m1"],p["m2"],p["m3"]
    w=gs.BoxWrap(verbose=False); w.start(psi); dt=model.opt.timestep
    trunk=[b for b in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or '').startswith('t1_u')]
    ts=set(trunk); out={}; n=0; bc0=data.xpos[gt.CUBE].copy()
    def meas():
        angs=[];cnt=0
        for i in range(data.ncon):
            c=data.contact[i]; b1=model.geom_bodyid[c.geom1]; b2=model.geom_bodyid[c.geom2]
            if (b1==gt.CUBE and b2 in ts) or (b2==gt.CUBE and b1 in ts):
                cnt+=1; d=c.pos-data.xpos[gt.CUBE]; angs.append(math.degrees(math.atan2(d[1],d[0])))
        cov=0
        if angs: cov=len(set(int(a//30) for a in angs))*30
        return cnt,cov
    while w.state=="running":
        data.ctrl[:]=0; w.step(dt); mujoco.mj_step(model,data); n+=1
        if w.phase=="HOLD" and "m" not in out and w.phase_t>0.7: out["m"]=meas()
    up=data.xmat[gt.CUBE].reshape(3,3)[2,2]; disp=np.linalg.norm(data.xpos[gt.CUBE][:2]-bc0[:2])
    return (x,y,psi,pose,out.get("m"),round(float(up),2),round(float(disp),3))
if __name__=="__main__":
    pose=(10,0,0.0,0.07)
    args=[(x,y,psi,pose) for x in (0.80,0.86) for y in (0.10,0.15) for psi in range(-180,180,30)]
    with Pool(2) as P:
        for r in P.imap(one,args): print(r,flush=True)
