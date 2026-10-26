"""Camera-only perception for the wheeled elephant robot.

The robot has no sensors. Everything it knows about the target box comes
from rendered camera images plus two pieces of prior knowledge a real robot
would also have:

  1. the box's size (7 x 7 x 20 cm, standing on the floor) and colour, and
  2. where its own cameras are on its own body (camera mounts + the head's
     neck angle), and that the floor is flat and the torso roughly level.

Method (monocular, no depth channel anywhere):

  * Render a camera, segment the box by colour, take the blob's bounding
    box in pixels.
  * The blob's BOTTOM edge is the box's near-face base, which lies on the
    floor: cast a ray through that pixel and intersect it with the floor
    plane -> a metric (x, y) position of the box, in the robot's own frame.
    The blob's TOP edge gives an independent estimate: it is the far top
    edge, 20 cm up, so intersect its ray with the z = 20 cm plane.
  * Box centre = that edge point pushed half a box width along the viewing
    direction (+3.5 cm from the near base, -3.5 cm from the far top edge).

All geometry is done in the robot's own TORSO frame (x forward, y left,
z up, floor at z = -base_height). Camera poses in that frame come from the
robot's own kinematics -- the same thing a CAD model plus the commanded
joint angles would give you -- NOT from where the torso is in the world.
World position, world orientation and the box's true pose are never read.

tip_cam rides on the flexible trunk, whose shape this robot has no way to
measure without joint sensors, so tip_cam is used for IMAGE-SPACE cues only
(is the box in view, where in the frame, how much of the frame it fills) --
never for metric geometry.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

import quadruped_grasp_core as core

model, data = core.model, core.data

# ---- prior knowledge about the target ---------------------------------
BOX_W = 0.035     # footprint (square) -- halved from 0.07
BOX_H = 0.20      # height
BOX_CENTER_Z = BOX_H / 2.0

import os
_res = os.environ.get("VISION_FRAME", "320x240").lower().split("x")
FRAME_W, FRAME_H = int(_res[0]), int(_res[1])   # override with VISION_FRAME=160x120 for slow software rendering

CAMS = {name: model.camera(name).id for name in
        ("trunk_cam", "chest_cam", "tip_cam", "eye_left", "eye_right")}

# ---- rendering ----------------------------------------------------------
_renderer = None
_renderer_failed = False


def _get_renderer():
    global _renderer, _renderer_failed
    if _renderer_failed:
        return None
    if _renderer is None:
        try:
            _renderer = mujoco.Renderer(model, height=FRAME_H, width=FRAME_W)
        except Exception as e:
            print(f"[VISION] offscreen renderer unavailable ({e!r}) -- the robot is "
                  f"camera-only, so it cannot run autonomously this session. Manual "
                  f"arrow-key driving still works; press X to abort.")
            _renderer_failed = True
            return None
    return _renderer


def renderer_ok():
    return _get_renderer() is not None


def render(cam_name):
    r = _get_renderer()
    if r is None:
        return None
    try:
        r.update_scene(data, camera=cam_name)
        # shadows/reflections don't help find a red box and cost ~6x render time
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        return r.render()
    except Exception as e:
        print(f"[VISION] render of {cam_name} failed ({e!r})")
        return None


# ---- colour segmentation --------------------------------------------------
def red_mask(img):
    """The box is red (cube_mat 0.75/0.15/0.15). Lit faces render ~ (190,38,38),
    shaded faces ~ (125,25,25); the orange trunk/wheels (high green) and the
    brown hide are rejected by the R >> G,B test."""
    R = img[..., 0].astype(np.int16)
    G = img[..., 1].astype(np.int16)
    B = img[..., 2].astype(np.int16)
    return (R > 70) & (R > 2.8 * G) & (R > 2.8 * B) & (np.abs(G - B) < 25)


@dataclass
class Blob:
    cam: str
    u: float          # centroid, pixels
    v: float
    u0: int           # bounding box (inclusive pixel indices)
    u1: int
    v0: int
    v1: int
    area: int
    clip_top: bool
    clip_bottom: bool
    clip_side: bool

    @property
    def fill(self):
        return self.area / float(FRAME_W * FRAME_H)

    @property
    def x_norm(self):   # -1 left edge .. +1 right edge
        return self.u / FRAME_W * 2.0 - 1.0

    @property
    def y_norm(self):   # -1 top .. +1 bottom
        return self.v / FRAME_H * 2.0 - 1.0


MIN_PIXELS = 12


def detect(cam_name, img=None):
    if img is None:
        img = render(cam_name)
    if img is None:
        return None
    mask = red_mask(img)
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PIXELS:
        return None
    v0, v1, u0, u1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    return Blob(cam_name, float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, u0, u1, v0, v1,
                int(len(xs)), v0 <= 0, v1 >= FRAME_H - 1, u0 <= 0 or u1 >= FRAME_W - 1)


# ---- the robot's own geometry (kinematics, not sensing) ----------------------
def _torso_frame():
    return data.xpos[core.torso_body].copy(), data.xmat[core.torso_body].reshape(3, 3).copy()


def base_height():
    """Height of the torso origin above the floor, from the robot's own leg
    geometry (wheel hubs below the torso + wheel radius). Changes when the
    legs crouch; independent of where the robot is in the world."""
    p_t, R_t = _torso_frame()
    hs = []
    for leg, bid in core.WHEEL_BODIES.items():
        hub = R_t.T @ (data.xpos[bid] - p_t)
        hs.append(-hub[2] + core.WHEEL_RADIUS)
    return float(np.mean(hs))


def camera_pose(cam_name):
    """Camera position and orientation in the TORSO frame (kinematics only)."""
    cid = CAMS[cam_name]
    p_t, R_t = _torso_frame()
    p = R_t.T @ (data.cam_xpos[cid] - p_t)
    R = R_t.T @ data.cam_xmat[cid].reshape(3, 3)
    return p, R


def _focal_px(cam_name):
    fovy = math.radians(float(model.cam_fovy[CAMS[cam_name]]))
    return (FRAME_H / 2.0) / math.tan(fovy / 2.0)


def pixel_ray(cam_name, u, v, pose=None):
    """Origin and unit direction (torso frame) of the ray through pixel (u, v).
    MuJoCo cameras look along their local -Z, with +X right and +Y up."""
    p, R = pose if pose is not None else camera_pose(cam_name)
    f = _focal_px(cam_name)
    d_cam = np.array([(u - FRAME_W / 2.0) / f, -(v - FRAME_H / 2.0) / f, -1.0])
    d = R @ d_cam
    return p, d / np.linalg.norm(d)


def _hit_plane(p, d, z):
    if d[2] > -0.02:      # ray not heading down toward the plane
        return None
    t = (z - p[2]) / d[2]
    if t <= 0:
        return None
    return p + t * d


def _unit_h(a):
    n = math.hypot(a[0], a[1])
    return np.array([a[0] / n, a[1] / n]) if n > 1e-9 else np.array([1.0, 0.0])


@dataclass
class BoxEstimate:
    x: float               # box centre, torso frame (forward)
    y: float               # (left)
    cam: str
    kind: str              # "base+top", "base", "top"
    spread: float          # |base-estimate - top-estimate| when both exist, else 0

    @property
    def range(self):       # horizontal distance from the torso origin
        return math.hypot(self.x, self.y)

    @property
    def bearing(self):     # + = box is to the robot's left
        return math.atan2(self.y, self.x)


def _close_range_fix(est):
    """Empirical correction for the chest camera at close range (x < 0.8 m): the
    ground-point model overestimates the lateral offset of an off-axis box by
    1.5-4 cm (it sees the near corner, not the face).  Fitted on a grid of box
    positions x 0.44-0.54, y 0.05-0.13 (any box yaw 0-45 deg): residual <= 1.1 cm."""
    if est is not None and est.cam == "chest_cam" and 0.3 < est.x < 0.8:
        est.y = 0.8481 * est.y + 0.1248 * est.x - 0.0706
    return est


def estimate_from_blob(blob, blob_img_mask=None):
    return _close_range_fix(_estimate_from_blob(blob, blob_img_mask))


def _estimate_from_blob(blob, blob_img_mask=None):
    """Metric box-centre estimate (torso frame) from one blob of a rigidly
    mounted camera (trunk_cam / chest_cam). None if the geometry degenerates."""
    if blob is None or blob.cam == "tip_cam":
        return None
    pose = camera_pose(blob.cam)
    zg = -base_height()
    est_b = est_t = None

    if not blob.clip_bottom:
        p, d = pixel_ray(blob.cam, blob.u, blob.v1 + 1.0, pose)   # bottom edge of last row
        P = _hit_plane(p, d, zg)
        if P is not None:
            est_b = P[:2] + (BOX_W / 2.0) * _unit_h(P[:2] - p[:2])
    if not blob.clip_top:
        p, d = pixel_ray(blob.cam, blob.u, blob.v0, pose)         # top edge of first row
        P = _hit_plane(p, d, zg + BOX_H)
        if P is not None:
            est_t = P[:2] - (BOX_W / 2.0) * _unit_h(P[:2] - p[:2])

    if est_b is not None and est_t is not None:
        spread = float(np.linalg.norm(est_b - est_t))
        # base edge is the more trustworthy (the floor is the best-known plane);
        # the top edge only refines it when the two agree
        if spread < 0.10:
            c, kind = 0.65 * est_b + 0.35 * est_t, "base+top"
        elif blob.cam == "trunk_cam":
            # the hanging trunk can cover the box's lower half in the head camera,
            # lifting the apparent base edge: the top edge is the trustworthy one
            c, kind = est_t, "top(base occluded?)"
        else:
            c, kind = est_b, "base(top disagrees)"
        return BoxEstimate(float(c[0]), float(c[1]), blob.cam, kind, spread)
    if est_b is not None:
        return BoxEstimate(float(est_b[0]), float(est_b[1]), blob.cam, "base", 0.0)
    if est_t is not None:
        return BoxEstimate(float(est_t[0]), float(est_t[1]), blob.cam, "top", 0.0)
    return None


def box_height_estimate(blob):
    """Apparent height (m) of the box from one rigidly-mounted camera, when both
    its base and top edges are in frame: the base edge fixes the horizontal
    distance (floor plane), then the top edge's ray height at that distance
    gives the top. A standing box reads ~0.20; one that has been knocked flat
    reads ~0.07. None if either edge is out of frame."""
    if blob is None or blob.cam == "tip_cam" or blob.clip_top or blob.clip_bottom:
        return None
    zg = -base_height()
    pose = camera_pose(blob.cam)
    p, db = pixel_ray(blob.cam, blob.u, blob.v1 + 1.0, pose)
    Pb = _hit_plane(p, db, zg)
    if Pb is None:
        return None
    r_base = math.hypot(Pb[0] - p[0], Pb[1] - p[1])
    _, dt = pixel_ray(blob.cam, blob.u, blob.v0, pose)
    dh = math.hypot(dt[0], dt[1])
    if dh < 1e-6:
        return None
    z_top = p[2] + (r_base + BOX_W) * dt[2] / dh      # far top edge, 1 box-width behind the near base
    return float(z_top - zg)


# which rigidly-mounted cameras give metric estimates, and over what range
# trunk_cam is pitched 40 deg down, so it loses the box beyond ~2.3 m; the two eye
# cameras (24 deg down, slightly verged) see it out to several metres. Searching /
# far-field uses all four, the final approach only needs the first two.
NEAR_CAMS = ("trunk_cam", "chest_cam")
FAR_CAMS = ("trunk_cam", "eye_left", "eye_right", "chest_cam")
METRIC_CAMS = NEAR_CAMS


def see(cams=METRIC_CAMS):
    """Render the metric cameras, return ({cam: Blob}, fused BoxEstimate or None)."""
    blobs, ests = {}, []
    for c in cams:
        b = detect(c)
        if b is None:
            continue
        blobs[c] = b
        e = estimate_from_blob(b)
        if e is not None:
            ests.append(e)
    return blobs, fuse(ests)


def fuse(ests):
    """Weighted fusion: closer-to-the-camera estimates are more accurate
    (pixel error maps to smaller metres), and a base+top pair is better than
    a single edge."""
    if not ests:
        return None
    if len(ests) >= 3:
        # one camera's view can be partly blocked by the trunk: drop estimates that
        # sit far from the median of all of them
        mx = float(np.median([e.x for e in ests]))
        my = float(np.median([e.y for e in ests]))
        kept = [e for e in ests if math.hypot(e.x - mx, e.y - my) < 0.12]
        ests = kept or ests
    ws, xs, ys, kinds, spread = [], [], [], [], 0.0
    for e in ests:
        p, _ = camera_pose(e.cam)
        dist = max(0.15, math.hypot(e.x - p[0], e.y - p[1]))
        w = (1.0 / dist ** 2) * (1.0 if e.kind == "base+top" else 0.6)
        ws.append(w); xs.append(e.x); ys.append(e.y); kinds.append(e.kind); spread = max(spread, e.spread)
    ws = np.array(ws)
    x = float(np.dot(ws, xs) / ws.sum())
    y = float(np.dot(ws, ys) / ws.sum())
    cams = "+".join(e.cam for e in ests)
    return BoxEstimate(x, y, cams, "/".join(sorted(set(kinds))), spread)


# ---- cable-pull direction from vision --------------------------------------
def psi_from_estimate(est, fallback=90.0):
    """The cable-pull azimuth (degrees, the angle cable_vec() expects) that
    points the trunk at the box, from the camera-derived box position. Same
    transform the old ground-truth version used -- box direction from the
    trunk mount, expressed in the trunk base frame, sign-flipped (see the old
    sensed_aim_local_dir) -- but fed by vision and done entirely in the torso
    frame."""
    if est is None:
        return fallback
    p_t, R_t = _torso_frame()
    mount = R_t.T @ (data.xpos[core.mount_body] - p_t)
    box = np.array([est.x, est.y, -base_height() + BOX_CENTER_Z])
    world_dir_in_torso = box - mount
    if np.linalg.norm(world_dir_in_torso) < 1e-9:
        return fallback
    R_tent = R_t.T @ data.xmat[core.tent_body].reshape(3, 3)
    lx, ly, _ = R_tent.T @ world_dir_in_torso
    lx, ly = -lx, -ly
    n = math.hypot(lx, ly)
    if n < 1e-9:
        return fallback
    return math.degrees(math.atan2(ly / n, lx / n))


# ---- image-space cues from the trunk-tip camera -----------------------------
def tip_view():
    """(Blob or None) from tip_cam. Image-space only -- see module docstring."""
    return detect("tip_cam")


# ---- visual lift estimate ------------------------------------------------------
def visual_lift(ref_est, blobs=None):
    """How far the box's bottom edge has risen off the floor, from chest_cam,
    given where the box was horizontally before the lift (ref_est, torso
    frame). The box's bottom edge at height h and horizontal distance r from
    the camera appears at depression angle atan((cam_z - zg - h) / r); invert
    for h. Returns metres, or None when the bottom edge isn't in view."""
    b = (blobs or {}).get("chest_cam") or detect("chest_cam")
    if b is None or b.clip_bottom:
        return None
    pose = camera_pose("chest_cam")
    p, d = pixel_ray("chest_cam", b.u, b.v1 + 1.0, pose)
    dh = math.hypot(d[0], d[1])
    if dh < 1e-6:
        return None
    zg = -base_height()
    u_h = _unit_h(d[:2])
    r = float(np.dot(np.array([ref_est.x, ref_est.y]) - p[:2], u_h)) - BOX_W / 2.0   # near face
    if r <= 0.05:
        return None
    tan_depr = -d[2] / dh
    z_bottom = p[2] - r * tan_depr
    return float(z_bottom - zg)
