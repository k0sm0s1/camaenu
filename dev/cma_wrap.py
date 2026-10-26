import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""TEST-ONLY CMA-ES over wrap parameters, scored on saved stack states (ground-truth scoring)."""
import sys, json, time, numpy as np, cma
from multiprocessing import Pool
SNAPS = ["a49", "a50", "a51", "a52", "b50", "b51"]
SPACE = [("nk",10,55),("tp",-85,-30),("td",0.03,0.12),("cr",0.02,0.12),("m1",5,30),("m2",5,30),("m3",15,30),
         ("psi",20,120),("t_up",3,8),("t_legs",3,8)]
DEF = dict(nk=30.93, tp=-76.92, td=0.0956, cr=0.0704, m1=10.59, m2=17.502, m3=30.0, psi=59.94, t_up=5.57, t_legs=4.09)
def enc(v,lo,hi): return float(np.arctanh(np.clip((v-lo)/(hi-lo)*2-1,-0.99,0.99)))
def decode(z): return {n: float(lo+(hi-lo)*(0.5+0.5*np.tanh(zi))) for (n,lo,hi),zi in zip(SPACE,z)}
def sc(r): return (3*r["final"]+2) if (r["up"]>0.8 and r["final"]>0.5) else (0.5*max(r["best"],0) if r["up"]>0.8 else 0.0)
def one(args):
    import feas3
    p, s = args
    try: return feas3.run(p, f"/tmp/snaps/{s}.npz")
    except Exception as e: return dict(final=-9,best=-9,up=0,minup=0)
def fit(z, pool):
    p = decode(z); rs = pool.map(one, [(p, s) for s in SNAPS]); ss = [sc(r) for r in rs]
    return -(np.mean(ss) + 0.5*min(ss)), [r["final"] for r in rs]
if __name__ == "__main__":
    ngen = int(sys.argv[1]); z0 = np.array([enc(DEF[n],lo,hi) for n,lo,hi in SPACE])
    es = cma.CMAEvolutionStrategy(z0, 0.3, {"popsize": 8, "seed": 5, "verbose": -9}); t0 = time.time(); best = (1e9, None, None)
    # evaluate the baseline first
    with Pool(2) as P:
        f0, r0 = fit(z0, P); print("baseline", round(f0,1), r0, flush=True); best = (f0, DEF, r0)
        for g in range(ngen):
            zs = es.ask(); out = [fit(z, P) for z in zs]; es.tell(zs, [o[0] for o in out])
            for z, o in zip(zs, out):
                if o[0] < best[0]: best = (o[0], decode(z), o[1])
            print(f"gen {g} best={best[0]:.1f} {best[2]} t={time.time()-t0:.0f}", flush=True)
            json.dump(dict(f=best[0], p=best[1], r=best[2]), open("/tmp/runs/cma_best.json", "w"), default=float)
