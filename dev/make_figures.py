import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "sim")); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "sim"))
"""Makes the README figures: what each camera sees, and the wrap phase-by-phase. Test/documentation only."""
import os, math, sys
import numpy as np, mujoco
from PIL import Image, ImageDraw
import quadruped_grasp_core as core, vision, grasp_sequence as gs, ground_truth_eval as gt, feas3
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images"
os.makedirs(OUT, exist_ok=True)
m, d = core.model, core.data

def label(img, text):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 22], fill=(20, 24, 30)); dr.text((6, 5), text, fill=(235, 235, 235))
    return np.array(im)

# ---- A. what each camera sees ------------------------------------------------------------
def cameras():
    mujoco.mj_resetDataKeyframe(m, d, 0); core.reset_commands(); gt.settle(800)
    gt.place_box(0.95, 0.12); gt.settle(400)
    tiles = []
    for cam in ("eye_left", "eye_right", "trunk_cam", "chest_cam", "chest_cam_mask", "overview"):
        real = cam.replace("_mask", "")
        img = vision.render(real)
        if img is None: continue
        b = vision.detect(real, img)
        if cam.endswith("_mask"):
            mk = vision.red_mask(img); out = np.zeros_like(img); out[mk] = (255, 255, 255); out[~mk] = (18, 22, 28)
            tiles.append(label(out, "chest_cam after the red-colour test (white = box pixels)")); continue
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        txt = cam
        if b is not None:
            dr.rectangle([b.u0, b.v0, b.u1, b.v1], outline=(255, 235, 0), width=2)
            txt += f"   box blob: {b.area} px, bottom edge at row {b.v1}"
        else:
            txt += "   box not in view"
        tiles.append(label(np.array(im), txt))
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    rows = []
    for r in range(0, len(tiles), 3):
        row = tiles[r:r+3]
        while len(row) < 3: row.append(np.zeros((h, w, 3), np.uint8))
        rows.append(np.concatenate([np.pad(t, ((0, h - t.shape[0]), (0, w - t.shape[1]), (0, 0))) for t in row], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(f"{OUT}/cameras.png")
    print("cameras done", flush=True)

# ---- B. the wrap, phase by phase -----------------------------------------------------------
def free_cam(renderer):
    """A camera that follows the torso: looks at the trunk and box from the robot's front-left."""
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    Rt = d.xmat[core.torso_body].reshape(3, 3); yaw = math.degrees(math.atan2(Rt[1, 0], Rt[0, 0]))
    cam.lookat[:] = d.xpos[core.torso_body] + Rt @ np.array([0.45, 0.08, -0.30]); cam.distance = 0.95
    cam.azimuth = yaw + 180 - 35; cam.elevation = -12
    return cam

def phases(snap="/tmp/snaps/a52.npz"):
    q = dict(feas3.DEF); q.update(nk=gs.TUNED["nk"], tp=gs.TUNED["tp"], td=gs.TUNED["td"], cr=gs.TUNED["cr"], m1=gs.TUNED["m1"],
             m2=gs.TUNED["m2"], m3=gs.TUNED["m3"], psi=gs.TUNED["psi"], t_up=gs.TUNED["t_up"], t_legs=gs.TUNED["t_legs"])
    S = np.load(snap); mujoco.mj_resetData(m, d)
    d.qpos[:] = S["qpos"]; d.qvel[:] = S["qvel"]; d.ctrl[:] = S["ctrl"]; d.act[:] = S["act"] if m.na else d.act
    d.qacc_warmstart[:] = S["warm"]; d.time = float(S["time"]); mujoco.mj_forward(m, d); core.current[:] = list(S["cur"])
    gs.SEQUENCE[:] = feas3.build(q); gs.STEP_PHASES.clear(); gs.M_SQUEEZE, gs.M_CINCH, gs.M_LIFT = q["m1"], q["m2"], q["m3"]
    wrap = gs.BoxWrap(verbose=False); wrap.start(q["psi"])
    R = mujoco.Renderer(m, height=480, width=640); cam = None
    want = {"REACH": "1  After the turn: trunk still up", "SQUEEZE": "2  Trunk lowered beside the box", "SETTLE1": "3  Cable squeeze: coil forms",
            "HOLD": "4  Cinched round the box", "HOLD2": "5  Trunk pulled up, box still down", "CARRY": "6  Legs straighten: box carried"}
    shots = {}; last = None; n = 0; dt = m.opt.timestep
    def grab(tag):
        cam = free_cam(R); R.update_scene(d, camera=cam); img = R.render()
        z = 100 * (d.xpos[gt.CUBE][2] - 0.1005); shots[tag] = label(img, f"{want[tag]}   (box height {z:+.1f} cm)")
    while wrap.state == "running":
        ph = wrap.phase
        if ph != last:
            if ph in want and ph not in shots: grab(ph)
            last = ph
        wrap.step(dt); mujoco.mj_step(m, d); n += 1
    grab("CARRY")
    keys = ["REACH", "SQUEEZE", "SETTLE1", "HOLD", "HOLD2", "CARRY"]
    tiles = [shots[k] for k in keys if k in shots]
    while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i+3], axis=1) for i in range(0, len(tiles), 3)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(f"{OUT}/wrap_phases.png")
    print("phases done", list(shots), flush=True)

if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "cameras"): cameras()
    if which in ("all", "phases"): phases()
