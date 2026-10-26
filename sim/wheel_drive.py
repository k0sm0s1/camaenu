"""Wheel-drive locomotion for the wheeled variant of the elephant
quadruped: replaces walk_gait.py's stepping-gait foot cycle with a
skid-steer wheel controller, per direct request ("keeping this like
this [the touch-trigger grasp pipeline] create another zip file where
we have the knees and we swap out the foot for wheels" -- full rolling
locomotion, confirmed directly, not a cosmetic-only foot-shape swap).

Why skid-steer, not real steering: the robot has no yaw-steering joint
(the hip abduction joint, j_{leg}_abd, only spreads the leg sideways
+-0.4363rad -- it was never a steering axis, and walk_gait.py's own
turn_bias already steers the ORIGINAL stepping gait the same way: by
making one side's stride longer than the other's, not by turning a
wheel/foot to point a new direction). Wheels make the equivalent trick
simpler, not different: drive the left-side wheels at one speed and the
right-side wheels at another, and the resulting thrust difference yaws
the whole robot -- same idea as a tank or a wheelchair, and the same
idea walk_gait.py already used for legs.

Sign convention matches walk_gait.py's SIDE_SIGN and turn_bias EXACTLY
on purpose: elephant_grasp_cv.py's VisionApproach (unchanged in this
package) drives (direction, turn_bias) without knowing whether legs or
wheels are on the other end, and manual UP/DOWN/LEFT/RIGHT in
autopilot_boxlift_cv.py already produces the same (direction, turn_bias)
values it always did. Reusing walk_gait.foot_target_manual's own
formula --
    stride = STRIDE * direction * (1 - turn_bias * SIDE_SIGN[leg])
-- as
    wheel_speed = BASE_ANGVEL * direction * (1 - turn_bias * SIDE_SIGN[leg])
means a positive turn_bias (LEFT key, or the camera steering left) slows
the LEFT wheels and speeds up the RIGHT wheels, same as it used to
shorten the left stride and lengthen the right one -- turning the same
way as before, with no changes needed anywhere upstream of this module.

Sign of "forward" was derived from the wheel joint's rotation axis
(0 1 0, same as the hip/knee -- see the XML comment on wheel_fl etc.)
and then DIRECTLY VERIFIED, not just derived: an isolated test drives
all four wheels at a constant positive BASE_ANGVEL and confirms the
torso's world-X position increases (the box sits at +X, so this must be
"forward" for the approach to work at all). See the roadmap for that
test's result; if a future model change ever flips this, that same test
is how to catch it again.
"""
import walk_gait

BASE_ANGVEL = 6.0   # rad/s commanded at full speed/no turn -- wheel radius
                     # is 0.05m (see the XML), so this is a first-cut
                     # ~0.30 m/s no-slip road speed. Not yet tuned against
                     # a measured net-displacement sweep the way
                     # walk_gait.py's STRIDE/X_OFFSET were for the old
                     # legs -- watch for slip (commanded speed higher than
                     # actual torso speed) before trusting this number for
                     # anything time-critical.


def wheel_targets(direction=1.0, turn_bias=0.0):
    """Returns {leg: angular_velocity_rad_s}. direction/turn_bias use the
    exact same range and sign as walk_gait.foot_target_manual did:
    direction=+1 forward/-1 backward, turn_bias in roughly [-1, 1],
    positive = turn left (matches core.held("LEFT") -> turn_bias=+0.4 in
    autopilot_boxlift_cv.py, and VisionApproach's own turn_bias sign,
    unchanged from the legged version)."""
    out = {}
    for leg in walk_gait.LEGS:
        out[leg] = BASE_ANGVEL * direction * (1.0 - turn_bias * walk_gait.SIDE_SIGN[leg])
    return out


SPIN_SPEED = 2.3   # x BASE_ANGVEL at rate=1.0 (=13.8 rad/s, just under the 14 rad/s actuator limit):
                   # ~14 deg/s in-place yaw measured on this model (slow: wheels must scrub sideways)


def spin_targets(rate=1.0):
    """Turn in place (skid-steer): left wheels one way, right wheels the other.
    rate > 0 turns LEFT (counter-clockwise from above), same sign convention
    as turn_bias. Returns {leg: angular_velocity_rad_s}."""
    out = {}
    for leg in walk_gait.LEGS:
        out[leg] = -BASE_ANGVEL * SPIN_SPEED * rate * walk_gait.SIDE_SIGN[leg]
    return out
