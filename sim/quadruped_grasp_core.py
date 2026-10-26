"""Core model loading, actuator helpers and keyboard input for the wheeled
elephant robot -- CAMERA-ONLY build.

This robot has NO sensors: no touch, no rangefinders, no IMU, no joint
sensors, no cable-force sensors (the model's <sensor> block is gone, and so
are the physical sensor bodies/sites it used). The only things that observe
the world are the cameras declared in robot_elephant_quadruped.xml:

    trunk_cam   head, face level          far-field search + approach
    chest_cam   torso front-bottom        close-range approach
    tip_cam     last trunk unit           eye-in-trunk, close range
    eye_left/eye_right  (spare)           head stereo pair, unused by default
    overview    world camera              for humans / debugging only

Nothing here reads the physics contact list or any object's true pose.
(Test-only ground truth lives in ground_truth_eval.py and is never imported
by the control code.)

What this module still reads from `data` is the robot's OWN body kinematics
(where its own links are, from its own commanded joint angles): see
vision.py's CameraRig for exactly where, and why that is not a sensor.
"""

import math
import os
import time
import ctypes

import mujoco
import numpy as np

import walk_gait

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(_SCRIPT_DIR, "robot_elephant_quadruped.xml")
if os.environ.get("ELEPHANT_MODEL"):
    MODEL_FILE = os.path.join(_SCRIPT_DIR, os.environ["ELEPHANT_MODEL"])

model = mujoco.MjModel.from_xml_path(MODEL_FILE)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
assert model.nsensor == 0, "camera-only build: the model must declare no <sensor>s"

# ---- bodies ----------------------------------------------------------
torso_body = model.body("torso").id
tent_body = model.body("tentacle_1").id      # trunk base frame (cable geometry lives here)
mount_body = tent_body                        # rigid mount point of the trunk
base_body = torso_body
WHEEL_BODIES = {leg: model.body(f"wheel_{leg}").id for leg in walk_gait.LEGS}
WHEEL_RADIUS = float(model.geom("g_fl_wheel").size[0])

# ---- actuators -------------------------------------------------------
T = [model.actuator(f"t1_pull_{i}").id for i in (1, 2, 3)]
# ---- trunk softness + cable force cap (hand-tunable without editing code) --------------------
# TRUNK_STIFF: stiffness multiplier at the trunk ROOT (default 0.25 = 4x softer). The scale fades
#   to 1.0 (unchanged) at the tip, because the tip joints are already ~40x softer than the root;
#   softening them as well would just make the tip curl before anything else moves. Joint damping
#   is scaled the same way so the trunk's time constants stay about the same.
# CABLE_MAX_N: force cap per cable (N). 60 was a simulation placeholder; ~30 N is what one
#   Dynamixel XL430-class servo on a 15 mm spool delivers comfortably.
TRUNK_STIFF = float(os.environ.get("TRUNK_STIFF", 0.25))
CABLE_MAX_N = float(os.environ.get("CABLE_MAX_N", 30.0))
_tj = sorted((j for j in range(model.njnt)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("t1_j")),
             key=lambda j: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
for _k, _j in enumerate(_tj):
    _s = TRUNK_STIFF ** (1.0 - _k / max(1, len(_tj) - 1))
    model.jnt_stiffness[_j] *= _s
    _d = model.jnt_dofadr[_j]
    model.dof_damping[_d:_d + 3] *= _s
for _a in T:
    model.actuator_ctrlrange[_a] = [0.0, CABLE_MAX_N]
    model.actuator_forcerange[_a] = [0.0, CABLE_MAX_N]

NECK_ACT = model.actuator("act_j_neck_pitch").id
TP_ACT = model.actuator("act_j_trunk_pitch").id
TD_ACT = model.actuator("act_j_trunk_drop").id
LEG_ACT = {leg: {"abd": model.actuator(f"act_j_{leg}_abd").id,
                 "hip": model.actuator(f"act_j_{leg}_hip").id,
                 "knee": model.actuator(f"act_j_{leg}_knee").id}
           for leg in walk_gait.LEGS}
WHEEL_ACT = {leg: model.actuator(f"act_j_{leg}_wheel").id for leg in walk_gait.LEGS}

STAND_TARGETS = {leg: (0.0, math.radians(21.57), math.radians(-43.13)) for leg in walk_gait.LEGS}


def apply_neck(angle):
    """Neck-pitch position servo, radians (0 = level, + = nod down). Call
    every tick: callers zero data.ctrl at the top of each step."""
    lo, hi = model.actuator_ctrlrange[NECK_ACT]
    data.ctrl[NECK_ACT] = float(np.clip(angle, lo, hi))


def apply_legs(abd_hip_knee_by_leg, dt=None):
    for leg in walk_gait.LEGS:
        abd, hip, knee = abd_hip_knee_by_leg[leg]
        data.ctrl[LEG_ACT[leg]["abd"]] = abd
        data.ctrl[LEG_ACT[leg]["hip"]] = hip
        data.ctrl[LEG_ACT[leg]["knee"]] = knee


def apply_wheels(wheel_vel_by_leg):
    """Per-wheel target angular velocity (rad/s); see wheel_drive.py."""
    for leg in walk_gait.LEGS:
        data.ctrl[WHEEL_ACT[leg]] = wheel_vel_by_leg[leg]


current = [0.0, 0.0, 0.0]   # last commanded cable tensions (rate-limited ramp state)


def apply_cables(target, dt, rate=6.0):
    """Rate-limited ramp of the three cable-pull commands toward `target`."""
    step = rate * dt
    for i in range(3):
        err = target[i] - current[i]
        if abs(err) <= step:
            current[i] = target[i]
        elif err > 0:
            current[i] += step
        else:
            current[i] -= step
        data.ctrl[T[i]] = current[i]


def reset_commands():
    """Forget the cable ramp state (use after resetting `data`)."""
    for i in range(3):
        current[i] = 0.0


class Debouncer:
    """True only once a boolean condition has held continuously for min_t
    seconds -- stops a single noisy camera frame from triggering anything."""

    def __init__(self, min_t):
        self.min_t = min_t
        self._since = None

    def update(self, condition, t):
        if not condition:
            self._since = None
            return False
        if self._since is None:
            self._since = t
        return (t - self._since) >= self.min_t


# ---------------------------------------------------------------------
# KEYBOARD (interactive viewer only). Three paths:
#   Windows : GetAsyncKeyState
#   viewer  : mujoco.viewer key_callback (macOS/Linux; needs no permissions)
#   macOS   : optional pynput fallback
user32 = None
try:
    user32 = ctypes.windll.user32
except AttributeError:
    pass

VK = {
    "U": 0x55, "O": 0x4F, "P": 0x50, "X": 0x58, "C": 0x43, "ESC": 0x1B,
    "LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28, "G": 0x47,
}

_mac_pressed = set()
_mac_listener = None
_MAC_SPECIAL = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT", "esc": "ESC"}


def _mac_key_name(key):
    char = getattr(key, "char", None)
    if char:
        return char.upper()
    return _MAC_SPECIAL.get(getattr(key, "name", ""))


def _ensure_mac_listener():
    global _mac_listener
    if _mac_listener is not None:
        return
    try:
        from pynput import keyboard
    except ImportError:
        _mac_listener = False
        return

    def on_press(key):
        name = _mac_key_name(key)
        if name:
            _mac_pressed.add(name)

    def on_release(key):
        name = _mac_key_name(key)
        if name:
            _mac_pressed.discard(name)

    _mac_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    _mac_listener.daemon = True
    _mac_listener.start()


# Viewer key callback path: pass viewer_key_callback to
# mujoco.viewer.launch_passive(..., key_callback=...). The callback only
# reports presses (plus OS key repeat), so each press counts as "held" for a
# short pulse -- arrow keys work as tap-or-hold.
_GLFW_KEYS = {71: "G", 88: "X", 67: "C", 85: "U", 79: "O", 80: "P",
              256: "ESC", 262: "RIGHT", 263: "LEFT", 264: "DOWN", 265: "UP"}
_cb_until = {}
# The viewer callback is the only key source by default. The pynput fallback
# starts an OS-level keyboard listener thread, which hard-crashes mjpython on
# macOS (no Python error) -- opt in with ELEPHANT_PYNPUT=1 if you really want it.
_cb_active = os.environ.get("ELEPHANT_PYNPUT") != "1"
KEY_PULSE_S = 0.2


def viewer_key_callback(keycode):
    global _cb_active
    name = _GLFW_KEYS.get(keycode)
    if name:
        _cb_active = True
        _cb_until[name] = time.time() + KEY_PULSE_S


def held(name):
    if user32 is not None:
        return bool(user32.GetAsyncKeyState(VK[name]) & 0x8000)
    if _cb_active and time.time() < _cb_until.get(name, 0.0):
        return True
    if _cb_active:
        return False
    _ensure_mac_listener()
    if _mac_listener is False:
        return False
    return name in _mac_pressed


_edge_state = {"C": False, "X": False, "G": False}


def pressed_edge(name):
    now = held(name)
    was = _edge_state.setdefault(name, False)
    _edge_state[name] = now
    return now and not was
