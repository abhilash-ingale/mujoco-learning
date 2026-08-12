"""Unitree G1 humanoid locomotion in MuJoCo — a tour of the whole toolchain.

    MJCF (XML)  --mj_compile-->  MjModel   constants: masses, geoms, actuators
                                    |
                                    v
                                 MjData    state: qpos, qvel, ctrl, contacts
                                    |
                controller --> d.ctrl --> mj_step() --> next MjData
                                    |
            +-------------+-------------+-------------+
            v             v             v             v
     mujoco.viewer   viser_bridge  mujoco.Renderer  nothing
     native window   browser/web    offscreen mp4   headless, fastest

Three controllers, in increasing order of ambition. All three are hand-written;
the numbers come from a random search over ~4000 simulated rollouts.

  stand  Hold a crouched pose with torso-attitude feedback.
         VERIFIED: upright for 60 s, drifts < 5 cm.

  march  Statically stable weight-shifting gait: 5.4 s per cycle, so the CoM is
         always over the stance foot. Genuinely lifts and replants each foot.
         VERIFIED: upright for 60 s -- but it steps IN PLACE, drifting only
         ~0.2 m backward over that minute. Stable, and going nowhere.

  walk   Dynamic gait: 1.6 s per cycle plus Raibert-style velocity feedback on
         foot placement. Reaches ~0.44 m/s -- a real walking speed.
         VERIFIED: takes about five steps, then FALLS at t ~ 4 s.

That last line is the point, not a bug. See "Why walk falls" in the README:
MuJoCo gives you physics, not control. Sustained dynamic humanoid walking needs
a learned policy, and `walk` is here to show you the wall you hit without one.

Run:
    mjpython examples/g1_walk.py --mode march        # viewer (macOS needs mjpython)
    python   examples/g1_walk.py --mode walk --viser  # browser at localhost:8080
    python   examples/g1_walk.py --mode walk --headless
    python   examples/g1_walk.py --mode stand --headless --duration 60
    python   examples/g1_walk.py --mode walk --video walk.mp4
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 1. THE MODEL
# ---------------------------------------------------------------------------
# `robot_descriptions` fetches and caches MuJoCo Menagerie, DeepMind's set of
# carefully tuned MJCF models. We load `scene.xml`, not `g1.xml`: the scene
# `<include>`s the robot and adds a floor, lights and a skybox. g1.xml on its
# own is a robot floating in a void with nothing to walk on.
from robot_descriptions.g1_mj_description import MJCF_PATH

SCENE_PATH = MJCF_PATH.replace("g1.xml", "scene.xml")

# The MJCF declares 29 `position` actuators (kp=500, dampratio=1), so a control
# value is a TARGET JOINT ANGLE in radians -- MuJoCo runs the PD law internally.
# With `motor` actuators instead, d.ctrl would be joint torques.
#
# Index bookkeeping, the part that bites everyone:
#   actuator i  <->  joint i  <->  qpos[7 + i]  <->  qvel[6 + i]
# The offsets are the floating base: qpos[0:3] xyz, qpos[3:7] quaternion (w x y z),
# qvel[0:3] linear velocity, qvel[3:6] angular velocity. That is why
# nq(36) = 7 + 29 but nv(35) = 6 + 29 -- a quaternion needs 4 numbers to store
# and only 3 to differentiate.
LEFT = dict(hip_pitch=0, hip_roll=1, hip_yaw=2, knee=3, ankle_pitch=4, ankle_roll=5)
RIGHT = dict(hip_pitch=6, hip_roll=7, hip_yaw=8, knee=9, ankle_pitch=10, ankle_roll=11)

FALL_HEIGHT = 0.45  # torso below this and it is on the floor (stands at 0.79)

# ---------------------------------------------------------------------------
# 2. THE GAITS
# ---------------------------------------------------------------------------
# crouch      nominal knee flexion; the pitch chain sums to zero so feet stay flat
# stride      hip-pitch sweep amplitude -> step length
# lift        extra knee flexion during swing -> foot clearance
# period      seconds per full cycle (two steps)
# duty        fraction of the cycle each leg spends in stance (>0.5 = double support)
# sway/_ph    lateral hip-roll weight shift, and its phase lead over the swing
# push        stance-leg hip extension that drives the body forward
# k_vx/k_vy   Raibert foot-placement feedback: step further out when moving faster
# kp_*/kd_*   torso attitude PD gains
GAITS = {
    "stand": dict(
        crouch=0.30, stride=0.0, lift=0.0, period=1.0, duty=0.75,
        sway=0.0, sway_ph=0.0, stance_kn=0.0, push=0.0,
        k_vx=0.0, k_vy=0.0, vdes=0.0,
        # Keep kp_p low. At kp_p=0.55 the pitch loop goes unstable and it
        # topples at t~3s: the hip correction rotates the torso further than the
        # error it was correcting. A classic too-much-gain oscillation.
        kp_p=0.20, kd_p=0.03, kp_r=0.50, kd_r=0.05, ankle_bias=0.0,
    ),
    # Slow enough to be statically stable: the CoM never leaves the stance foot.
    "march": dict(
        crouch=0.2961, stride=0.0989, lift=0.2923, period=5.3564, duty=0.7907,
        sway=0.1351, sway_ph=0.4875, stance_kn=0.068, push=0.0,
        k_vx=0.0, k_vy=0.0, vdes=0.0,
        kp_p=0.1509, kd_p=0.0, kp_r=1.3315, kd_r=0.0589, ankle_bias=0.002,
    ),
    # Fast enough to be genuinely dynamic -- and therefore to fall over.
    "walk": dict(
        crouch=0.3168, stride=0.1033, lift=0.1287, period=1.6184, duty=0.729,
        sway=-0.30, sway_ph=1.0, stance_kn=0.1177, push=0.0017,
        k_vx=0.3975, k_vy=-0.2156, vdes=0.4,
        kp_p=0.1899, kd_p=0.0296, kp_r=1.2267, kd_r=0.056, ankle_bias=-0.15,
    ),
}


def nominal_pose(model: mujoco.MjModel, g: dict) -> np.ndarray:
    """Control vector for the crouched neutral stance.

    Starts from the model's built-in `stand` keyframe, which is where the arm
    pose comes from -- let the model author choose it for you -- then overrides
    the legs. hip_pitch + knee + ankle_pitch = 0 keeps the soles flat.
    """
    ctrl = model.key_ctrl[0].copy()
    for side in (LEFT, RIGHT):
        ctrl[side["hip_pitch"]] = -g["crouch"] / 2
        ctrl[side["knee"]] = g["crouch"]
        ctrl[side["ankle_pitch"]] = -g["crouch"] / 2
    return ctrl


def torso_attitude(data: mujoco.MjData) -> tuple[float, float, float, float]:
    """Roll, pitch and their rates, read straight out of MjData."""
    w, x, y, z = data.qpos[3:7]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch, data.qvel[3], data.qvel[4]


def control(t: float, data: mujoco.MjData, g: dict, base: np.ndarray,
            lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map (time, state) -> 29 target joint angles.

    This function is the entire "policy". Replace it with a neural network and
    you have the modern RL pipeline; every other line in this file stays put.
    """
    ctrl = base.copy()
    roll, pitch, roll_rate, pitch_rate = torso_attitude(data)

    # Feedback that keeps the torso vertical. Without this even `stand` topples.
    pitch_corr = g["kp_p"] * pitch + g["kd_p"] * pitch_rate
    roll_corr = g["kp_r"] * roll + g["kd_r"] * roll_rate

    # Raibert heuristic: if the body is travelling faster than desired, plant the
    # swing foot further forward to brake; if slower, plant it short to accelerate.
    place_x = g["k_vx"] * (data.qvel[0] - g["vdes"])
    place_y = g["k_vy"] * data.qvel[1]

    phase = (t / g["period"]) % 1.0
    for side, offset in ((LEFT, 0.0), (RIGHT, 0.5)):  # legs half a cycle apart
        a = (phase + offset) % 1.0
        if a < g["duty"]:  # ---- STANCE: foot planted, sweep the hip backward ----
            u = a / g["duty"]
            hip = g["stride"] * (0.5 - u) * 2 - g["push"] * np.sin(np.pi * u)
            knee = g["crouch"] + g["stance_kn"]
        else:  # ---- SWING: foot in the air, swing forward and lift the knee ----
            u = (a - g["duty"]) / (1 - g["duty"])
            hip = g["stride"] * (u - 0.5) * 2 + place_x
            knee = g["crouch"] + g["lift"] * np.sin(np.pi * u)

        hip = hip - g["crouch"] / 2 + pitch_corr
        ankle = -(hip + knee) + g["ankle_bias"]  # keep the sole level
        sway = g["sway"] * np.sin(2 * np.pi * (phase + g["sway_ph"]))

        ctrl[side["hip_pitch"]] = hip
        ctrl[side["knee"]] = knee
        ctrl[side["ankle_pitch"]] = ankle
        ctrl[side["hip_roll"]] = sway - roll_corr + place_y
        ctrl[side["ankle_roll"]] = -sway + roll_corr

    # Never command past the limits the MJCF declares -- the servo would just
    # fight the joint stop and waste torque.
    return np.clip(ctrl, lo, hi)


# ---------------------------------------------------------------------------
# 3. THE LOOP
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="G1 locomotion in MuJoCo")
    ap.add_argument("--mode", choices=tuple(GAITS), default="march")
    ap.add_argument("--duration", type=float, default=20.0, help="sim seconds")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--video", help="write an mp4 (needs imageio + imageio-ffmpeg)")
    ap.add_argument("--viser", action="store_true", help="stream to a browser")
    ap.add_argument("--port", type=int, default=8080, help="viser port")
    ap.add_argument("--collision", action="store_true",
                    help="with --viser, also draw collision geoms")
    ap.add_argument("--loop", action="store_true",
                    help="with --viser, restart when the run ends")
    ap.add_argument("--follow", action="store_true",
                    help="with --viser, keep the camera centred on the robot")
    args = ap.parse_args()

    g = GAITS[args.mode]

    # --- compile MJCF -> MjModel (constant), then allocate MjData (state) ---
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    # Keyframes set qpos, qvel and ctrl in one call -- the clean way to start.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    base = nominal_pose(model, g)
    data.ctrl[:] = base
    settle = 0.8  # let it sink into the crouch before the gait starts

    print(f"mode={args.mode}  nq={model.nq} nv={model.nv} nu={model.nu} "
          f"dt={model.opt.timestep}  period={g['period']}s")

    fell_at: float | None = None

    def reset() -> None:
        nonlocal fell_at
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = base
        fell_at = None

    def step() -> None:
        """One physics step. Write ctrl, call mj_step. That is the whole API."""
        nonlocal fell_at
        t = data.time - settle
        data.ctrl[:] = control(t, data, g, base, lo, hi) if t > 0 else base
        mujoco.mj_step(model, data)  # integrate dynamics + solve contacts, one dt
        if fell_at is None and data.qpos[2] < FALL_HEIGHT:
            fell_at = data.time
            print(f"  !! fell at t={fell_at:.2f}s after {data.qpos[0]:+.2f} m")

    if args.viser:
        run_viser(model, data, step, reset, args)
    elif args.video:
        run_video(model, data, step, args)
    elif args.headless:
        run_headless(model, data, step, args)
    else:
        run_viewer(model, data, step, args)

    x, y, z, t = data.qpos[0], data.qpos[1], data.qpos[2], data.time
    print(f"final: t={t:.2f}s  x={x:+.2f}m  y={y:+.2f}m  torso_z={z:.2f}m  "
          f"mean_speed={x / max(t, 1e-9):.3f} m/s  "
          f"{'UPRIGHT' if z > FALL_HEIGHT else f'FELL at {fell_at:.2f}s'}")


def run_headless(model, data, step, args) -> None:
    """No rendering at all. The mode you tune and train in: ~30-50x realtime."""
    wall = time.perf_counter()
    while data.time < args.duration:
        step()
    elapsed = time.perf_counter() - wall
    print(f"{int(args.duration / model.opt.timestep)} steps in {elapsed:.2f}s "
          f"-> {args.duration / elapsed:.0f}x realtime")


def run_viewer(model, data, step, args) -> None:
    """Interactive window. On macOS this needs `mjpython`, not `python`."""
    import mujoco.viewer

    n = 10  # physics steps between GUI syncs: 500 Hz sim, ~50 Hz render
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.5, -15, 120
        while viewer.is_running() and data.time < args.duration:
            tick = time.perf_counter()
            for _ in range(n):
                step()
            viewer.cam.lookat[:] = data.qpos[:3]  # track the robot
            viewer.sync()  # hand the new MjData to the render thread
            lag = n * model.opt.timestep - (time.perf_counter() - tick)
            if lag > 0:
                time.sleep(lag)  # run at wall-clock speed, not flat out


def run_viser(model, data, step, reset, args) -> None:
    """Stream to a browser. Works headless and over SSH, unlike mujoco.viewer.

    Structurally identical to run_viewer: step physics n times, push one frame,
    sleep to hold wall-clock pace. Only the sink differs.
    """
    from viser_bridge import ViserBridge

    bridge = ViserBridge(model, data, port=args.port,
                         show_collision=args.collision, follow=args.follow)
    print("viser: waiting for a browser to connect...")
    if not bridge.wait_for_client():
        print("viser: nobody connected, running anyway")

    n = 10  # physics steps per pushed frame -> ~50 Hz over the websocket
    while True:
        tick = time.perf_counter()
        for _ in range(n):
            step()
        bridge.sync()
        lag = n * model.opt.timestep - (time.perf_counter() - tick)
        if lag > 0:
            time.sleep(lag)
        if data.time >= args.duration:
            if not args.loop:
                break
            reset()


def run_video(model, data, step, args) -> None:
    """Offscreen: MjData -> MjvScene -> pixels -> mp4."""
    import imageio.v2 as imageio

    fps = 30
    per_frame = max(1, round(1 / (fps * model.opt.timestep)))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 3.5, -15, 120
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(args.video, fps=fps) as writer:
            while data.time < args.duration:
                for _ in range(per_frame):
                    step()
                cam.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, cam)
                writer.append_data(renderer.render())
    print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
