"""Camera-only autonomous stack for the wheeled elephant robot.

    SEARCH -> APPROACH -> (ALIGN) -> SETTLE -> WRAP -> VERIFY -> DONE
                 \\______ any state --> FAILED / abort (X) ______/

Inputs: camera images only (trunk_cam, chest_cam, and tip_cam for image-space
cues) via vision.py. Outputs: wheel speeds, neck angle, and -- through
grasp_sequence.BoxWrap -- the trunk/cable/crouch commands. No sensor, contact
query or true object pose is read anywhere in this file.

What each state does
  SEARCH    spin slowly in place until either metric camera sees the box.
  APPROACH  visual servoing on the box's camera-derived position (torso
            frame): steer on bearing, drive on (forward distance - PARK_X),
            slow down as it closes. Estimates are low-pass filtered and
            jump-gated; brief dropouts coast on the last command.
  ALIGN     close enough but not square-on: turn in place until bearing ~ 0,
            nudge forward/back to hit PARK_X.
  SETTLE    stop, let the swaying trunk damp out, average the camera estimate
            over a short window, and only then commit.
  WRAP      the fixed choreography (grasp_sequence.BoxWrap) with the cable
            azimuth psi computed from the camera estimate. Meanwhile the
            cameras watch the box: if it visibly gets pushed away the wrap
            is aborted and the trunk relaxed (the vision-only replacement
            for the old "a leg/trunk touched the box" contact checks).
  VERIFY    after the wrap, chest_cam estimates how high the box's bottom
            edge is above the floor (visual lift) and tip_cam reports
            whether the box is still in front of the trunk tip. Only a
            camera-confirmed lift of >= LIFT_OK counts as success; anything
            else ends in FAILED with the reason (never a silent 'done').

Honest limits are in the project README (the wrap choreography itself).
"""
import math
import os
import time

import numpy as np
import mujoco

import quadruped_grasp_core as core
import wheel_drive
import vision
import grasp_sequence as gs

model, data = core.model, core.data

# ---- tuning ----------------------------------------------------------------
PARK_X = float(os.environ.get("PARK_X", 0.52))   # box-centre distance ahead of the torso centre when it stops (m)
# The trunk points straight ahead, so a box dead ahead gets the trunk laid on top of it.  Instead
# the robot stops TURNED by PARK_ANGLE degrees: the box sits that many degrees off the robot's
# nose (positive = box on the LEFT), the trunk comes down BESIDE it, then curls round it.
PARK_ANGLE = float(os.environ.get("PARK_ANGLE", 20.0))                           # deg off the nose
PARK_Y = float(os.environ.get("PARK_Y", PARK_X * math.tan(math.radians(PARK_ANGLE))))   # sideways offset (m, + = left)
PARK_BEARING = math.atan2(PARK_Y, PARK_X)   # aim the box at this bearing, not dead ahead
PARK_X_TOL = 0.010
# after the roll-forward, yaw toward the box until it is this many degrees off the nose (same side
# as PARK_ANGLE). That turn sweeps the lowered trunk in from the side. WRAP_TURN=0 disables it.
CONTACT_X = float(os.environ.get("WRAP_CONTACT_X", 0.50))   # camera fix: box this close (torso frame) = trunk is touching it
CONTACT_Y = float(os.environ.get("WRAP_CONTACT_Y", 0.10))   # ... and this far to the side
LATERAL = float(os.environ.get("WRAP_LATERAL", 0.055))   # m: sideways offset the box must have when the robot then drives STRAIGHT to it
ADV_TIMEOUT = float(os.environ.get("WRAP_ADV_TIMEOUT", 12.0))   # s, safety stop for the creep forward
TURN_BEARING = math.radians(float(os.environ.get("WRAP_TURN_BEARING", 10.5))) * (1.0 if PARK_ANGLE >= 0 else -1.0)
DO_TURN = os.environ.get("WRAP_TURN", "1") != "0"
PARK_BEARING_TOL = 0.022  # rad (~1.3 deg)
K_YAW = 1.6              # turn_bias per radian of bearing error
TURN_IN_PLACE_ABOVE = 0.30   # rad: bigger bearing errors are removed by spinning first (turning radius while
TURN_IN_PLACE_BELOW = 0.12   # driving is ~1.5 m on this robot), until the error is back under this
MAX_TURN = 0.6
V_FAR, V_NEAR = 1.0, 0.22   # wheel "direction" command far away / near the park spot
SLOW_ZONE = 0.5          # start slowing this far (m) before PARK_X
RENDER_EVERY = int(os.environ.get("STACK_RENDER_EVERY", "15"))   # physics ticks between camera frames (15 = ~33 Hz at 500 Hz)
SEARCH_RATE = 1.0        # spin-in-place rate while searching (~14 deg/s)
SEARCH_TIMEOUT = 40.0    # s -- a full turn takes ~26 s at SEARCH_RATE
LOST_COAST_T = 0.6       # s to keep the last command after the box drops out of view
LOST_GIVEUP_T = 4.0
SETTLE_T = 1.0           # s stopped before committing
SETTLE_AVG_T = 0.4       # s of estimates averaged at the end of SETTLE
JUMP_GATE = 0.35         # m: reject single-frame estimate jumps larger than this
ALPHA = 0.6              # low-pass weight on the newest estimate
DISPLACE_ABORT = float(os.environ.get("STACK_DISPLACE_ABORT", "0.20"))    # m the box may move during the wrap before we call it knocked
WRAP_WATCH_EVERY = 100   # ticks between cameras checks during the wrap (~5 Hz)
MAX_SETTLE_RETRIES = 3
FAR_CAM_ABOVE = 1.8      # m: beyond this (or when the box is lost) the eye cameras are used too
LIFT_OK = 0.01           # m: the camera must see the box's bottom edge at least this far off the floor
FLAT_HEIGHT = 0.15       # m: a box the cameras measure shorter than this has been knocked over


class AutonomousStack:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.wrap = gs.BoxWrap(verbose=verbose)
        self.state = "idle"
        self.reason = ""
        self._reset()

    # ---- bookkeeping -------------------------------------------------------
    def _reset(self):
        self.t = 0.0
        self.state_t = 0.0
        self._tick = 0
        self.est = None             # filtered box estimate (torso frame) or None
        self.est_age = 0.0
        self.blobs = {}
        self.turn_bias = 0.0
        self.direction = 0.0
        self.spin = 0.0
        self._ok_frames = 0
        self._turning = False
        self._settle_samples = []
        self._retries = 0
        self.ref_est = None         # box estimate at the moment the wrap started
        self.psi = None
        self._adv_t = 0.0
        self._turn_t = 0.0
        self._touch_ok = 0
        self._adv_last = None
        self._turn_best = 9.0; self._turn_best_t = 0.0
        self._turn_ok = 0
        self.visual_lift_m = None
        self.tip_fill = None
        self._lift_hist = []
        self._displaced_frames = 0
        self._flat_frames = 0
        self.log = []

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _go(self, new_state, why=""):
        self.state = new_state
        self.state_t = 0.0
        self._log(f"[STACK] -> {new_state}" + (f"  ({why})" if why else ""))

    @property
    def active(self):
        return self.state not in ("idle", "done", "failed")

    # ---- commands ------------------------------------------------------------
    def start(self):
        if not vision.renderer_ok():
            self.state, self.reason = "failed", "offscreen renderer unavailable"
            return False
        self._reset()
        self.wrap = gs.BoxWrap(verbose=self.verbose)
        self._go("search", "looking for the box")
        return True

    def abort(self, why="aborted"):
        if self.state != "idle":
            self._log(f"[STACK] abort ({why})")
        self.wrap.abort()
        self.state, self.reason = "idle", why
        self.direction = self.turn_bias = self.spin = 0.0

    def fail(self, why):
        self._log(f"[STACK] FAILED: {why}")
        self.wrap.abort()
        self.state, self.reason = "failed", why
        self.direction = self.turn_bias = self.spin = 0.0

    # ---- perception ---------------------------------------------------------------
    def _perceive(self, dt_frame):
        far = self.state == "search" or self.est is None or self.est.x > FAR_CAM_ABOVE or self.est_age > 0.5
        if self.state == "wrap":
            blobs, raw = vision.see(("chest_cam",))   # torso-fixed: the head cameras move with the neck during the wrap and bias the fix
        else:
            blobs, raw = vision.see(vision.FAR_CAMS if far else vision.NEAR_CAMS)
        self.blobs = blobs
        if raw is None:
            self.est_age += dt_frame
            return None
        if self.est is not None and self.est_age < 1.0:
            jump = math.hypot(raw.x - self.est.x, raw.y - self.est.y)
            if jump > JUMP_GATE:
                self.est_age += dt_frame      # single-frame outlier: ignore it
                return None
            raw.x = ALPHA * raw.x + (1 - ALPHA) * self.est.x
            raw.y = ALPHA * raw.y + (1 - ALPHA) * self.est.y
        self.est, self.est_age = raw, 0.0
        return raw

    # ---- one physics tick ------------------------------------------------------------
    def step(self, dt):
        """Call once per physics tick while `active`. Writes data.ctrl."""
        if not self.active:
            return
        self.t += dt
        self.state_t += dt
        self._tick += 1
        frame_dt = dt * RENDER_EVERY

        if self.state == "wrap":
            self._step_wrap(dt)
            return
        if self.state == "verify":
            self.step_verify(dt)
            return

        # every non-wrap state holds the stance and the neck level
        core.apply_legs(core.STAND_TARGETS)
        core.apply_neck(0.0)
        core.apply_cables([0.0, 0.0, 0.0], dt)

        if self._tick % RENDER_EVERY == 0:
            self._perceive(frame_dt)
            self._decide(frame_dt)

        # wheels run every tick from the latest decision
        if self.spin != 0.0:
            core.apply_wheels(wheel_drive.spin_targets(self.spin))
        else:
            core.apply_wheels(wheel_drive.wheel_targets(self.direction, self.turn_bias))

    # ---- decision logic (once per camera frame) -----------------------------------------
    def _decide(self, frame_dt):
        e = self.est if (self.est is not None and self.est_age < 0.2) else None

        if self.state == "search":
            self.direction = self.turn_bias = 0.0
            self.spin = SEARCH_RATE
            if e is not None:
                self.spin = 0.0
                self._go("approach", f"box seen: x={e.x:.2f} y={e.y:+.2f} via {e.cam}")
            elif self.state_t > SEARCH_TIMEOUT:
                self.fail("box not found after a full turn")
            return

        if self.state == "approach":
            self.spin = 0.0
            if e is None:
                # coast briefly on the last command, then stop and look around
                if self.est_age > LOST_GIVEUP_T:
                    self.direction = self.turn_bias = 0.0
                    self._go("search", "box lost")
                elif self.est_age > LOST_COAST_T:
                    self.direction = min(self.direction, V_NEAR) * 0.5
                    self.turn_bias = 0.0
                return
            bearing = e.bearing - PARK_BEARING
            to_go = e.x - PARK_X
            if abs(bearing) > TURN_IN_PLACE_ABOVE:
                self._turning = True
            elif abs(bearing) < TURN_IN_PLACE_BELOW:
                self._turning = False
            if self._turning:                       # big heading error: square up first
                self.direction = self.turn_bias = 0.0
                self.spin = float(np.clip(2.5 * bearing, -1.0, 1.0))
                if abs(self.spin) < 0.25:
                    self.spin = 0.25 * math.copysign(1.0, self.spin)
                return
            self.turn_bias = float(np.clip(K_YAW * bearing, -MAX_TURN, MAX_TURN))
            self.direction = float(np.clip(
                V_NEAR + (V_FAR - V_NEAR) * (to_go / SLOW_ZONE), V_NEAR, V_FAR)) if to_go > 0 else 0.0
            if to_go <= PARK_X_TOL:
                self.direction = self.turn_bias = 0.0
                self._go("align", f"at park range (x={e.x:.2f}), squaring up")
            return

        if self.state == "align":
            if e is None:
                if self.est_age > LOST_GIVEUP_T:
                    self._go("search", "box lost while aligning")
                return
            to_go = e.x - PARK_X
            good_b = abs(e.bearing - PARK_BEARING) <= PARK_BEARING_TOL
            good_x = abs(to_go) <= PARK_X_TOL
            self.turn_bias = 0.0
            if not good_b:
                self.direction = 0.0
                self.spin = float(np.clip(2.0 * (e.bearing - PARK_BEARING), -0.5, 0.5))
                if abs(self.spin) < 0.2:
                    self.spin = 0.2 * math.copysign(1.0, self.spin)
            elif not good_x:
                self.spin = 0.0
                self.direction = float(np.clip(1.5 * to_go, -V_NEAR, V_NEAR))
                if abs(self.direction) < 0.1:
                    self.direction = 0.1 * math.copysign(1.0, self.direction)
            else:
                self.spin = self.direction = 0.0
                self._ok_frames += 1
                if self._ok_frames >= 3:
                    self._ok_frames = 0
                    self._settle_samples = []
                    self._go("settle", f"aligned: x={e.x:.3f} y={e.y:+.3f}")
            if not (good_b and good_x):
                self._ok_frames = 0
            return

        if self.state == "settle":
            self.direction = self.turn_bias = self.spin = 0.0
            if self.state_t >= SETTLE_T - SETTLE_AVG_T and e is not None:
                self._settle_samples.append((e.x, e.y))
            if self.state_t >= SETTLE_T:
                if len(self._settle_samples) < 3:
                    self._retries += 1
                    if self._retries > MAX_SETTLE_RETRIES:
                        return self.fail("could not get a stable camera fix at the park spot")
                    return self._go("align", "no stable fix yet")
                xs, ys = zip(*self._settle_samples)
                fx, fy = float(np.mean(xs)), float(np.mean(ys))
                if abs(fx - PARK_X) > 3 * PARK_X_TOL or abs(math.atan2(fy, fx) - PARK_BEARING) > 3 * PARK_BEARING_TOL:
                    self._retries += 1
                    if self._retries > MAX_SETTLE_RETRIES:
                        return self.fail("could not hold the park position")
                    return self._go("align", f"drifted off the spot (x={fx:.2f}, y={fy:+.2f})")
                self.ref_est = vision.BoxEstimate(fx, fy, self.est.cam if self.est else "?", "settled", 0.0)
                self.psi = gs.PSI_FALLBACK * (1.0 if PARK_Y >= 0 else -1.0)   # curl TOWARD the box (mirrored if it is on the right)
                tv = vision.tip_view()
                self.tip_fill = None if tv is None else tv.fill
                self._log(f"[STACK] camera fix at wrap start: x={fx:.3f} y={fy:+.3f}  psi={self.psi:.1f} deg  "
                          f"tip_cam sees box: {'yes (fill %.1f%%)' % (100 * tv.fill) if tv else 'no'}")
                self._adv_t = self._turn_t = 0.0; self._touch_ok = 0
                self._adv_last = None; self._turn_best = 9.0; self._turn_best_t = 0.0
                self._turn_ok = 0
                self.wrap.hold_phases = set()
                if gs.TUNED["adv"] > 0:
                    self.wrap.hold_phases.add("ADVANCE")   # optional forward creep (default off)
                if DO_TURN:
                    self.wrap.hold_phases.add("TURN")
                self.est, self.est_age = None, 9.0     # drop the head-camera fix; the wrap uses chest_cam only
                self.wrap.start(self.psi)
                self._displaced_frames = 0
                self._go("wrap")
            return

    # ---- wrap + watch ---------------------------------------------------------------------
    def _advance_command(self, dt):
        """ADVANCE phase: the box is ~TURN_BEARING off the nose, the trunk hangs passive beside it.
        Creep the ROBOT forward until the box is at the trunk (est.x <= CONTACT_X). Near the end the
        box slips out of chest_cam's view, so the last camera fix is carried forward with the
        commanded drive speed. Returns (direction, turn_bias)."""
        self._adv_t += dt
        fresh = self.est is not None and self.est_age < 0.3
        speed = 0.258 * gs.ADV_DIR
        if fresh:
            self._adv_last = (self.est.x, self.est.y, self._adv_t)
        pred = None
        if self._adv_last is not None:
            lx, ly, lt = self._adv_last
            pred = lx - speed * (self._adv_t - lt)
        touching = pred is not None and pred <= CONTACT_X and abs(self._adv_last[1]) <= CONTACT_Y + 0.05
        done_by = None
        if touching:
            src = "camera" if fresh else "last camera fix + drive distance"
            done_by = f"trunk at the box ({src}: box ~{pred:.2f} m ahead)"
        elif self._adv_t >= ADV_TIMEOUT:
            done_by = "timeout"
        elif self._adv_last is None and self._adv_t >= 3.0:
            done_by = "no camera fix at all"
        if done_by:
            self.wrap.release_phase("ADVANCE")
            self._log(f"[STACK] advance done in {self._adv_t:.1f} s -- {done_by}")
            return 0.0, 0.0
        return gs.ADV_DIR, 0.0

    def _turn_command(self, dt):
        """TURN phase: yaw (in place) until the camera sees the box TURN_BEARING off the nose.
        Returns the spin rate (+ = left)."""
        self._turn_t += dt
        fresh = self.est is not None and self.est_age < 0.3
        if self._turn_t < 0.3:
            fresh = False         # wait for a fix taken after the reach finished
        # driving straight keeps the sideways offset constant, so aim the turn at THAT offset
        err = (self.est.bearing - math.copysign(math.atan2(LATERAL, max(self.est.x, 0.3)), TURN_BEARING)) if fresh else None
        self._turn_ok = self._turn_ok + 1 if (fresh and abs(err) < 0.025) else 0
        # contact cue: the robot keeps spinning but the box's bearing stops closing -> the trunk is on it
        if fresh:
            if self.est.bearing < self._turn_best - 0.006:
                self._turn_best, self._turn_best_t = self.est.bearing, self._turn_t
            stalled = (self._turn_t > 1.5 and self._turn_t - self._turn_best_t > 0.8
                       and abs(self.est.y) < LATERAL + 0.06)
        else:
            stalled = False
        done_by = None
        if stalled:
            done_by = (f"camera: bearing stopped closing -- trunk touching the box "
                       f"({self.est.x:.2f} m ahead, {self.est.y:+.2f} m left)")
        elif self._turn_ok >= 3:
            done_by = f"camera: box now {self.est.x:.2f} m ahead, {self.est.y:+.2f} m left ({math.degrees(self.est.bearing):.0f} deg off the nose)"
        elif self._turn_t > 10.0:
            done_by = "timeout"
        elif not fresh and self._turn_t > 3.0:
            done_by = "no camera fix"
        if done_by:
            self.wrap.release_phase("TURN")
            self._log(f"[STACK] turn done in {self._turn_t:.1f} s -- {done_by}")
            return 0.0
        if err is None:
            return 0.0
        rate = float(np.clip(2.0 * err, -0.5, 0.5))
        return rate if abs(rate) >= 0.2 else 0.2 * math.copysign(1.0, rate)   # slow creep so it touches, not slams

    def _step_wrap(self, dt):
        if self._tick % RENDER_EVERY == 0:
            self._perceive(dt * RENDER_EVERY)      # keeps the box estimate live (advance feedback + telemetry)
        spin = 0.0
        if self.wrap.phase == "ADVANCE" and "ADVANCE" in self.wrap.hold_phases:
            d_cmd, t_cmd = self._advance_command(dt)
        elif self.wrap.phase == "TURN" and "TURN" in self.wrap.hold_phases:
            d_cmd, t_cmd = 0.0, 0.0
            spin = self._turn_command(dt)
        else:
            d_cmd, t_cmd = 0.0, 0.0
        if spin != 0.0:
            core.apply_wheels(wheel_drive.spin_targets(spin))
        else:
            core.apply_wheels(wheel_drive.wheel_targets(d_cmd, t_cmd))
        self.wrap.step(dt)
        if self.wrap.state == "holding":
            self._go("verify")

    def verify_step(self):
        """One camera look at the held box: (visual lift in m or None, tip_cam fill or None)."""
        lift = vision.visual_lift(self.ref_est) if self.ref_est else None
        tv = vision.tip_view()
        return lift, (None if tv is None else tv.fill)

    def step_verify(self, dt):
        """Hold the carry pose, take several camera looks over ~1 s, then judge
        by the MEDIAN (a single look can be fooled by the trunk crossing the
        box's base or by the box still swinging)."""
        core.apply_wheels(wheel_drive.wheel_targets(0.0, 0.0))
        self.wrap.step(dt)            # state 'holding' keeps re-applying the final pose
        if self.state_t >= 0.5 and self._tick % WRAP_WATCH_EVERY == 0:
            lift, fill = self.verify_step()
            self._lift_hist.append((lift, fill))
        if self.state_t >= 1.6:
            lifts = [l for l, _ in self._lift_hist if l is not None]
            fills = [f for _, f in self._lift_hist if f is not None]
            lift = float(np.median(lifts)) if lifts else None
            fill = float(np.median(fills)) if fills else None
            self.visual_lift_m, self.tip_fill = lift, fill
            msg = ("lift (camera, median of %d looks) = %.1f cm" % (len(lifts), lift * 100)) if lift is not None \
                else "lift: box base not visible"
            msg += ", tip_cam box fill = %s" % ("%.1f%%" % (100 * fill) if fill is not None else "box not in tip view")
            self._log(f"[STACK] VERIFY: {msg}")
            if lift is None or lift < LIFT_OK:
                return self.fail("grasp not confirmed by camera -- " + msg)
            self._go("done", msg)
