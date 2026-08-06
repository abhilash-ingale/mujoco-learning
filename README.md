# mujoco-learning

Notes and runnable examples for learning [MuJoCo](https://mujoco.readthedocs.io),
DeepMind's physics engine for robotics and control. The worked example is a
Unitree G1 humanoid, used as a vehicle for walking through the entire toolchain
from XML model to rendered video.

---

## Installation

macOS (Apple Silicon). MuJoCo ships native arm64 wheels, so there is nothing to
compile.

### 1. Python 3.12

MuJoCo itself works on 3.13/3.14, but the surrounding robotics ecosystem
(`gymnasium`, `dm_control`, JAX/MJX) lags a release or two behind. 3.12 is the
sweet spot.

```bash
brew install python@3.12
```

### 2. Virtual environment

```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

### 3. Dependencies

```bash
pip install --upgrade pip && pip install mujoco robot_descriptions imageio imageio-ffmpeg
```

| Package | Why |
| --- | --- |
| `mujoco` | The engine plus its Python bindings, viewer and renderer |
| `robot_descriptions` | Downloads and caches robot models (MuJoCo Menagerie, URDFs) |
| `imageio` + `imageio-ffmpeg` | Optional — only needed to write mp4 files |

`glfw`, `PyOpenGL` and `numpy` arrive automatically as `mujoco` dependencies.

### 4. Verify

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Confirmed working on this machine: **Python 3.12.13, mujoco 3.11.0, arm64**.

### macOS gotcha: `mjpython`

The interactive viewer needs the process's **main thread** to own the Cocoa
window, which `python` does not give it. MuJoCo ships a launcher that does:

```bash
mjpython examples/g1_walk.py
```

Use plain `python` for everything headless — batch simulation, tuning, RL
training, offscreen mp4 rendering. Only the live GUI needs `mjpython`. Getting
this wrong produces `launch_passive requires that the Python script is run under
mjpython on macOS`.

---

## Running the example

```bash
mjpython examples/g1_walk.py --mode march
```

```bash
python examples/g1_walk.py --mode walk --headless --duration 10
```

```bash
python examples/g1_walk.py --mode walk --video walk.mp4
```

[`examples/g1_walk.py`](examples/g1_walk.py) implements three hand-written
controllers, tuned by random search over roughly 4,000 simulated rollouts. The
measured results, all reproducible with the commands above:

| Mode | What it does | Outcome |
| --- | --- | --- |
| `stand` | Holds a crouched pose with torso-attitude feedback | Upright 60 s, drifts 3 cm |
| `march` | Statically stable weight shift, 5.4 s per cycle | Upright 60 s, but steps **in place** |
| `walk` | Dynamic gait, 1.6 s per cycle + velocity feedback | Reaches 0.44 m/s, then **falls at t≈4.1 s** |

That last row is the honest headline, and it is discussed in
[Why `walk` falls over](#why-walk-falls-over) below.

---

## The MuJoCo toolchain, via this example

### MJCF → `MjModel`: the robot description

MuJoCo models are XML in a dialect called **MJCF**. You almost never write one
for a real robot from scratch; you fetch a tuned one. `robot_descriptions`
handles the download and caching:

```python
from robot_descriptions.g1_mj_description import MJCF_PATH
# ~/.cache/robot_descriptions/mujoco_menagerie/unitree_g1/g1.xml
```

The example loads `scene.xml`, **not** `g1.xml`. This distinction matters and is
a standard Menagerie convention:

- `g1.xml` — the robot alone: bodies, joints, geoms, actuators. No floor, no
  lights. Load this and the robot falls forever through an empty void.
- `scene.xml` — `<include>`s `g1.xml` and adds the ground plane, a skybox, a
  headlight and visual settings. This is what you actually simulate.

Compiling XML into the engine's internal form gives you an **`MjModel`**:

```python
model = mujoco.MjModel.from_xml_path(SCENE_PATH)   # or from_xml_string(...)
```

`MjModel` is everything **constant** about the world: masses, inertias, geom
shapes, joint axes and limits, actuator gains, the integrator timestep. You read
it constantly and you rarely mutate it. For the G1:

```
nq=36   position coordinates
nv=35   velocity coordinates
nu=29   actuators
```

### The `nq` ≠ `nv` trap

`nq=36` but `nv=35`, and the one-off will corrupt your indexing if you assume
they match. The G1 has a **floating base** — a free joint connecting the pelvis
to the world:

| | `qpos` | `qvel` |
| --- | --- | --- |
| Base position | `[0:3]` — x, y, z | `[0:3]` — linear velocity |
| Base orientation | `[3:7]` — quaternion (w, x, y, z) | `[3:6]` — angular velocity |
| Joints | `[7:36]` — 29 angles | `[6:35]` — 29 rates |

A 3-D rotation takes **4** numbers to store as a unit quaternion but only **3**
to differentiate. Hence `7 + 29 = 36` versus `6 + 29 = 35`. The practical
consequence, and the line worth memorising:

```
actuator i  <->  joint i  <->  qpos[7 + i]  <->  qvel[6 + i]
```

### `MjData`: the state

```python
data = mujoco.MjData(model)
```

`MjData` is everything that **changes**: `qpos`, `qvel`, `ctrl`, contact lists,
computed Jacobians, sensor readings, `data.time`. One `MjModel` can back many
`MjData` instances — that is exactly how vectorised RL runs thousands of
parallel environments off a single compiled model.

Rather than hand-assembling a start pose, use a **keyframe** the model author
already tuned. The G1 MJCF defines one named `stand`:

```python
mujoco.mj_resetDataKeyframe(model, data, 0)   # sets qpos, qvel AND ctrl at once
```

The example reads `model.key_ctrl[0]` to inherit the author's arm pose, then
overrides only the legs — worth copying as a habit.

### Actuators: what `ctrl` actually means

This is model-dependent and the single most common source of confusion. The G1
MJCF declares:

```xml
<position kp="500" dampratio="1" inheritrange="1"/>
```

These are **position servos**, so `data.ctrl[i]` is a **target joint angle in
radians** and MuJoCo runs the PD law internally at every substep. Had the model
used `<motor>`, the very same array would mean **joint torques**. Always check
before writing to `ctrl`; the units are not implied by the API.

The MJCF also declares per-joint `range` and `actuatorfrcrange`. Commanding past
a joint limit just makes the servo fight the joint stop and burn torque, so the
example clips every command:

```python
np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
```

### The step loop

The core contract of the entire engine is two lines:

```python
data.ctrl[:] = my_controller(data)   # decide
mujoco.mj_step(model, data)          # integrate dynamics + solve contacts, one dt
```

`model.opt.timestep` is 0.002 s for the G1, i.e. **500 Hz** physics. `mj_step`
does forward kinematics, computes the mass matrix, solves the constraint/contact
problem, and integrates.

Useful relatives:

- `mj_forward` — recompute all derived quantities **without** advancing time.
  Use it after manually setting `qpos` so contacts and sensors are consistent.
- `mj_step1` / `mj_step2` — split the step, letting a controller see the
  post-kinematics state before forces are applied.
- `mj_resetData` — back to the model defaults.

Control almost never wants 500 Hz. Real policies run at 20–50 Hz and hold their
output across many physics steps — that ratio is a genuine hyperparameter, not
an implementation detail.

### Three ways to see it

The example implements all three, and they share the identical `step()` function
— rendering is strictly downstream of physics.

**1. Headless** — no graphics. The mode you tune, test, and train in. Measured
at **29–48× realtime** for the G1 on this machine, single-threaded (falling
robots are slower — more contacts for the solver to resolve).

```python
while data.time < duration:
    step()
```

**2. Interactive viewer** — `launch_passive` returns control to *your* loop, so
you own the clock (`launch` instead hands the loop to MuJoCo). "Passive" means
the viewer only observes; it never steps physics for you.

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for _ in range(10):        # 500 Hz physics, ~50 Hz GUI
            step()
        viewer.cam.lookat[:] = data.qpos[:3]   # track the robot
        viewer.sync()              # hand new MjData to the render thread
```

Two details the example handles and beginners usually miss: `sync()` every
single physics step wastes most of your frame budget, and without an explicit
`time.sleep` the simulation runs 25× too fast to watch.

**3. Offscreen renderer** — `MjData` → `MjvScene` → pixels, for mp4 or for
pixel-observation RL. Needs no window, so plain `python` is fine.

```python
with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data, cam)
    frame = renderer.render()      # (480, 640, 3) uint8 numpy array
```

### Where the controller sits

Everything above is fixed infrastructure. The *only* interesting line is:

```python
data.ctrl[:] = control(t, data, ...)
```

In `g1_walk.py`, `control()` is a central pattern generator: a phase clock
splits each leg's cycle into **stance** (foot planted, hip sweeping backward)
and **swing** (foot lifted, hip swinging forward), offset half a cycle between
legs, plus a `duty` factor above 0.5 so both feet overlap in double support. On
top sit two feedback terms:

- **Torso attitude PD** on roll and pitch, read off the base quaternion.
- **Raibert foot placement** — step further forward when travelling faster than
  desired. The standard trick for stabilising legged velocity.

Swap this one function for a neural network and you have the modern RL pipeline.
Nothing else in the file changes.

---

## Why `walk` falls over

Worth stating plainly, because it is the most useful thing in this repo: **a
hand-tuned controller did not get the G1 walking dynamically, and that is the
expected result.**

The search covered four progressively richer controller parameterisations —
plain CPG, explicit stance/swing with a duty factor, signed lateral sway phase,
and Raibert velocity feedback — over ~4,000 rollouts of random search plus local
refinement. The outcome was a hard trade-off with nothing in between:

- Optimise for **survival** and you get `march`: upright indefinitely, stepping
  in place at ~0.01 m/s. Statically stable, and useless as locomotion.
- Optimise for **distance** and you get `walk`: a credible 0.44 m/s for about
  five steps, then a fall at t≈4.1 s. Every fast gait the search found died in
  2–4 s.

The reason is structural. A 29-DoF humanoid at walking speed is an
**underactuated, hybrid** system: it changes dynamics every time a foot makes or
breaks contact, the feet can only push (never pull), and balance depends on
where the *next* footstep lands rather than on any instantaneous joint error. An
open-loop clock with a couple of PD terms has no mechanism to reason about that.
Adding gain does not help — it actively hurts, which the `stand` gains
demonstrate concretely: at `kp_p=0.55` the pitch loop goes unstable and topples
a *standing* robot in 3 s, while `kp_p=0.20` holds it for a minute.

The two approaches that do work, neither of which is a small addition:

1. **Model-based** — ZMP/capture-point footstep planning with a preview
   controller, whole-body inverse kinematics, and an ankle/hip stabiliser. This
   is how most pre-2018 humanoids walked.
2. **Reinforcement learning** — the current standard and how the real G1 walks.
   [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
   ships `G1JoystickFlatTerrain`, a JAX/MJX environment that trains a robust
   velocity-tracking policy in minutes on a GPU. The policy is a small MLP
   consuming the same `qpos`/`qvel` this example reads and emitting the same 29
   position targets — it drops into `control()` unchanged.

MuJoCo gives you physics, not control. `walk` is in this repo to make the shape
of that gap concrete rather than to hide it.

---

## Reference

- [MuJoCo docs](https://mujoco.readthedocs.io) — the XML reference and
  computation chapters are the ones worth reading properly
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — tuned models
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — RL environments
- [`robot_descriptions.py`](https://github.com/robot-descriptions/robot_descriptions.py) — model loader
