import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
import sys, json, time, numpy as np, cma
from multiprocessing import Pool
SPACE=[("x",0.62,0.78),("y",-0.04,0.14),("psi",0,170),("nk",0,55),("tp",-80,30),("td",0.03,0.12),("cr",0.0,0.12),
       ("m1",8,50),("m2",20,60),("m3",30,60),("t_legs",3,10),("t_up",3,10)]
def enc(v,lo,hi): return float(np.arctanh(np.clip((v-lo)/(hi-lo)*2-1,-0.99,0.99)))
def decode(z): return {n: lo+(hi-lo)*(0.5+0.5*np.tanh(zi)) for (n,lo,hi),zi in zip(SPACE,z)}
def fit(z):
    import feas
    p=decode(z)
    try: r=feas.run(dict(p,t_reach=4.0))
    except Exception as e: return 5.0,None
    up=r["upright"]>0.9
    f=(r["final_lift_cm"]*3 if up else 0.0)+(0.2*max(r["max_lift_cm"],0))+(2 if up else 0)
    return -f, {k:round(float(v),2) for k,v in r.items()}
if __name__=="__main__":
    seed=json.load(open("/tmp/wrap2_best.json"))["p"]; seed.update(m3=60,t_legs=5.0,t_up=6.0)
    z0=np.array([enc(seed[n],lo,hi) for n,lo,hi in SPACE])
    es=cma.CMAEvolutionStrategy(z0,0.5,{"popsize":12,"seed":11,"verbose":-9}); P=Pool(2); best=(0,None,None); t=time.time()
    for g in range(int(sys.argv[1])):
        zs=es.ask(); out=P.map(fit,zs); es.tell(zs,[o[0] for o in out])
        for z,o in zip(zs,out):
            if o[0]<best[0]: best=(o[0],decode(z),o[1])
        print(f"gen {g} best={best[0]:.1f} {best[2]} t={time.time()-t:.0f}",flush=True)
        json.dump(dict(f=best[0],p=best[1],r=best[2]),open("/tmp/lift2_best.json","w"),default=float)
