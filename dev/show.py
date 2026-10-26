import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
import json,sys
for l in open(sys.argv[1]):
    cfg,_,r=l.partition(' => ')
    try: j=json.loads(r)
    except Exception: print(cfg,'ERR',r[:150]); continue
    print(f"{cfg:60s} final {j['final_lift']:6.2f} up {j['up']:5.2f} best {j['best_up_lift']:5.2f} minup {j['min_up']:5.2f} {j['state']} | R{j['phases'].get('REACH')} S{j['phases'].get('SQUEEZE')}")
