import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
import sys, json, time, numpy as np, cma, math
from multiprocessing import Pool
SPACE=[("x",0.62,0.78),("y",-0.04,0.14),("psi",0,170),("nk",0,50),("tp",-80,30),("td",0.0,0.12),("cr",0.0,0.1),("m1",8,50),("m2",20,60)]
def decode(z): return {n: lo+(hi-lo)*(0.5+0.5*np.tanh(zi)) for (n,lo,hi),zi in zip(SPACE,z)}
def one(p):
    import quadruped_grasp_core as core, grasp_sequence as gs, ground_truth_eval as gt, feas, mujoco
    model,data=core.model,core.data
    q=dict(p,t_reach=4.0,m3=60)
    mujoco.mj_resetDataKeyframe(model,data,0); core.reset_commands(); gt.settle(800); gt.place_box(p["x"],p["y"]); gt.settle(200)
    gs.SEQUENCE[:]=feas.build(q)[:5]; gs.STEP_PHASES.clear(); gs.M_SQUEEZE,gs.M_CINCH,gs.M_LIFT=q["m1"],q["m2"],60
    w=gs.BoxWrap(verbose=False); w.start(q["psi"]); dt=model.opt.timestep
    ts=set(b for b in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or '').startswith('t1_u'))
    bc0=data.xpos[gt.CUBE].copy(); best=(0,0)
    while w.state=="running":
        data.ctrl[:]=0; w.step(dt); mujoco.mj_step(model,data)
        if w.phase in("CINCH","HOLD") and int(w.phase_t/dt)%200==0:
            angs=[]
            for i in range(data.ncon):
                c=data.contact[i]; b1=model.geom_bodyid[c.geom1]; b2=model.geom_bodyid[c.geom2]
                if (b1==gt.CUBE and b2 in ts) or (b2==gt.CUBE and b1 in ts):
                    d=c.pos-data.xpos[gt.CUBE]; angs.append(math.degrees(math.atan2(d[1],d[0])))
            cov=len(set(int(a//30) for a in angs))*30
            if cov>best[0]: best=(cov,len(angs))
    up=data.xmat[gt.CUBE].reshape(3,3)[2,2]; disp=float(np.linalg.norm(data.xpos[gt.CUBE][:2]-bc0[:2]))
    return best,float(up),disp
def fit(z):
    p=decode(z)
    try: (cov,n),up,disp=one(p)
    except Exception as e: return 0.0,None
    ok= up>0.9 and disp<0.06
    return -((cov+min(n,10)+20) if ok else 0.15*cov), (cov,n,round(up,2),round(disp,3))
if __name__=="__main__":
    es=cma.CMAEvolutionStrategy(np.zeros(len(SPACE)),0.7,{"popsize":12,"seed":7,"verbose":-9}); P=Pool(2); best=(0,None,None); t=time.time()
    for g in range(int(sys.argv[1])):
        zs=es.ask(); out=P.map(fit,zs); es.tell(zs,[o[0] for o in out])
        for z,o in zip(zs,out):
            if o[0]<best[0]: best=(o[0],decode(z),o[1])
        print(f"gen {g} best={best[0]:.0f} {best[2]} t={time.time()-t:.0f}",flush=True)
        json.dump(dict(f=best[0],p=best[1],r=best[2]),open("/tmp/wrap2_best.json","w"),default=float)
