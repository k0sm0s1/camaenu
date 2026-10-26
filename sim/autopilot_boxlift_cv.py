"""Interactive MuJoCo viewer for the CAMERA-ONLY wheeled elephant.

  python autopilot_boxlift_cv.py        (macOS: mjpython autopilot_boxlift_cv.py)

The robot has no sensors at all -- only cameras (trunk_cam, chest_cam,
tip_cam). Press G and it runs the whole job from images alone:

    search for the red box -> drive up to it -> square up -> settle ->
    wrap the trunk around it -> check the result by camera

Controls
  G             start the autonomous run (search / approach / wrap / verify)
  X             abort: stop everything, relax the trunk, back to manual
  UP / DOWN     drive forward / backward   (manual; only when idle)
  LEFT / RIGHT  turn while driving         (manual; only when idle)
  ESC           quit

Terminal output narrates what the cameras see (box position estimates, the
state machine, the camera-derived cable angle) once a second.

See elephant_grasp_cv.py for the stack, vision.py for how the cameras are
turned into metric box positions, and the README for the honest status.
"""
import time

import mujoco
import mujoco.viewer

import quadruped_grasp_core as core
import wheel_drive
import elephant_grasp_cv as cv

model, data = core.model, core.data


def main():
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    stack = cv.AutonomousStack()
    dt = model.opt.timestep
    last_report = 0.0

    import grasp_sequence as _gs
    print(f"[autopilot] park: box {cv.PARK_X:.2f} m ahead, {cv.PARK_Y:+.2f} m left (= {cv.PARK_ANGLE:.0f} deg off the nose) | wrap: " + " ".join(f"{k}={v:g}" for k, v in _gs.TUNED.items()))
    print("[autopilot] camera-only robot.  G=run autonomously  X=abort  "
          "UP/DOWN/LEFT/RIGHT=drive by hand  ESC=quit")

    with mujoco.viewer.launch_passive(
            model, data, key_callback=core.viewer_key_callback) as viewer:
        # chase camera: behind and above the torso, so "forward" reads as away from you
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = core.base_body
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20

        while viewer.is_running():
            data.ctrl[:] = 0.0

            if core.held("ESC"):
                break

            if core.pressed_edge("X"):
                stack.abort("X pressed")

            if core.pressed_edge("G") and not stack.active:
                stack.start()

            if stack.active:
                stack.step(dt)
                if stack.t - last_report >= 1.0:
                    last_report = stack.t
                    e = stack.est
                    seen = "box NOT in view" if e is None or stack.est_age > 0.3 else (
                        f"box at x={e.x:.2f} m ahead, y={e.y:+.2f} m left  "
                        f"(range {e.range:.2f} m, bearing {e.bearing * 57.3:+.0f} deg, via {e.cam})")
                    print(f"[autopilot] {stack.state:8s} {seen}")
            elif stack.wrap.state in ("running", "holding"):
                # finished (or aborted mid-way into 'holding'): keep the carry pose
                core.apply_wheels(wheel_drive.wheel_targets(0.0, 0.0))
                stack.wrap.step(dt)
            else:
                direction = 0.0
                if core.held("UP"):
                    direction = 1.0
                elif core.held("DOWN"):
                    direction = -1.0
                turn_bias = 0.0
                if core.held("LEFT"):
                    turn_bias = 0.4
                elif core.held("RIGHT"):
                    turn_bias = -0.4
                core.apply_neck(0.0)
                core.apply_legs(core.STAND_TARGETS)
                core.apply_wheels(wheel_drive.wheel_targets(direction, turn_bias))
                core.apply_cables([0.0, 0.0, 0.0], dt)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
