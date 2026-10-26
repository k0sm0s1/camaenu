import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""Continuation search: start from the lifting wrap (box 0.497 m out) and move the park farther out
(0.56-0.66 m), scoring the WORSE of {exact park, park off by (+1 cm x, -1 cm y)} so it must be forgiving."""
import sys, json, time, numpy as np, cma
from multiprocessing import Pool
SPACE=[("x",0.56,0.66),("y",-0.02,0.14),("psi",0,170),("nk",0,55),("tp",-80,30),("td",0.03,0.12),("cr",0.0,0.12),
       ("m1",8,50),("m2",20,60),("m3",30,60),("t_legs",3,10),("t_up",3,10)]
def enc(v,lo,hi): return float(np.arctanh(np.clip((v-lo)/(hi-lo)*2-1,-0.99,0.99)))
def decode(z): return {n: lo+(hi-lo)*(0.5+0.5*np.tanh(zi)) for (n,lo,hi),zi in zip(SPACE,z)}
def score(r):
    up=r["upright"]>0.85
    return (r["final_lift_cm"]*3 if up else 0.0)+(0.2*max(r["max_lift_cm"],0))+(2 if up else 0)
def fit(z):
    import feas
    p=dict(decode(z),t_reach=4.0)
    try:
        r1=feas.run(p); q=dict(p); q["x"]+=0.01; q["y"]-=0.01; r2=feas.run(q)
    except Exception: return 5.0,None
    s=min(score(r1),score(r2))
    return -s, {"nom":round(float(r1["final_lift_cm"]),1),"off":round(float(r2["final_lift_cm"]),1),"up":round(float(r1["upright"]),2)}
if __name__=="__main__":
    seed=json.load(open("best_wrap_params.json"))["p"]; seed["x"]=0.56
    z0=np.array([enc(seed[n],lo,hi) for n,lo,hi in SPACE])
    es=cma.CMAEvolutionStrategy(z0,0.35,{"popsize":12,"seed":21,"verbose":-9}); P=Pool(2); best=(0,None,None); t=time.time()
    for g in range(int(sys.argv[1])):
        zs=es.ask(); out=P.map(fit,zs); es.tell(zs,[o[0] for o in out])
        for z,o in zip(zs,out):
            if o[0]<best[0]: best=(o[0],decode(z),o[1])
        print(f"gen {g} best={best[0]:.1f} {best[2]} t={time.time()-t:.0f}",flush=True)
        json.dump(dict(f=best[0],p=best[1],r=best[2]),open("/tmp/lift3_best.json","w"),default=float)
