import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
import sys, json, time, numpy as np, cma
from multiprocessing import Pool
# name, lo, hi
SPACE=[("x",0.40,0.95),("y",-0.10,0.10),("psi",-180,180),("nk",20,55),("tp",-80,40),
       ("td",0.0,0.12),("cr",0.0,0.12),("t_reach",2.0,8.0),("m1",5,40),("m2",20,80),("m3",30,100)]
def decode(z):
    return {n: lo+(hi-lo)*(0.5+0.5*np.tanh(zi)) for (n,lo,hi),zi in zip(SPACE,z)}
def fit(z):
    import feas
    p=decode(z); 
    try: r=feas.run(p)
    except Exception as e: return 5.0, {"err":str(e)}
    up=r["upright"]>0.9
    f=(r["final_lift_cm"] if up else 0.0)+0.3*max(r["max_lift_cm"],0)+(0.5 if up else 0.0)
    return -f, r
if __name__=="__main__":
    import feas
    z0=np.zeros(len(SPACE)); 
    es=cma.CMAEvolutionStrategy(z0,0.8,{"popsize":12,"seed":int(sys.argv[1]) if len(sys.argv)>1 else 1,"verbose":-9})
    best=(0,None,None); pool=Pool(2); t=time.time()
    for g in range(int(sys.argv[2]) if len(sys.argv)>2 else 30):
        zs=es.ask(); out=pool.map(fit,zs); es.tell(zs,[o[0] for o in out])
        for z,(f,r) in zip(zs,out):
            if f<best[0]: best=(f,decode(z),r)
        print(f"gen {g} best_f={best[0]:.2f} gen_min={min(o[0] for o in out):.2f} t={time.time()-t:.0f}s best={ {k:round(v,2) for k,v in (best[1] or {}).items()} } res={best[2]}",flush=True)
        json.dump(dict(best=best[1],res=best[2]),open("/tmp/feas_best.json","w"),default=float)
