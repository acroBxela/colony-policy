# Colony Can-Am policy (`colony_exp_limited_ft`)

[Trajectory rollouts](plots/trajectory_rollouts.png) ·
[States and controls, including ay](plots/states_and_controls_vs_s.png)

## Nominal trajectory for trajopt_drive

[Download the complete bundle](nominal_trajectory.zip), or browse
[nominal_trajectory/](nominal_trajectory/). It contains `traj.csv`,
`surface_track.npz`, `meta.yaml`, `stats.json`, and native plotting previews.
This is a rollout of this exact published policy, starting at vx=0.1 m/s,
s=e=beta=0, previous torque=250 Nm: 1030 samples, one lap in 20.5632 s.
Startup is retained; only the final crossing is interpolated to exact lap-end s.
The bundle was validated using drive_trajopt's loader and plotting code.

In an environment with drive_trajopt and its dependencies installed:

```python
from drive_trajopt.traj import load_artifact
trajectory = load_artifact("nominal_trajectory")
```

```bash
python -m drive_trajopt.traj nominal_trajectory
```

Use active track coordinates (`s_track`, `e_track`, `dphi_track`). The surface
is the original track corridor, not a fitted or independently optimized path.
See the bundle's README and metadata for command timing and coordinate details.

Self-contained **CPU JAX/StableHLO inference artifact**, not a dynamics model,
standalone executable, or real-car safety controller. No imports from the RL
project or the vehicle-model repository are needed at inference time.
This standalone repository contains exported inference functions and loaders,
not the XAL source code, raw training checkpoints, or training data. The unroll
artifact also embeds the frozen dynamics needed for prediction.

## Load and run

Use a separate environment (tested with Python 3.11 and JAX 0.7.2):

```bash
python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu python load_policy.py
```

From this directory:

```python
import jax.numpy as jnp
from load_policy import load_policy

policy = load_policy()
measured = jnp.array([
    x, y, yaw, vx, vy, yaw_rate, rear_wheel_speed,
    accel_x, accel_y, measured_steer, measured_torque,
], dtype=jnp.float32)
previous = jnp.array([previous_steer_command, previous_torque_command], dtype=jnp.float32)
command = policy(measured, previous, jnp.float32(0.02))
steer_command, torque_command = command.tolist()
previous = command  # Retain for the next tick.
```

Warm up and synchronize once before real-time inference. Load only trusted
artifacts. Caller supplies finite values, positive dt, and in-bounds previous
commands. The script sends nothing to a vehicle. Target-machine timing and
hardware compatibility have not been validated.

## Contract and coordinates

`float32[11], float32[2], float32 scalar -> float32[2]`.
Inputs and outputs are physical SI units, not normalized values:

- x/y: meters east/north in Colony's surveyed local ENU frame, with geodetic
  origin latitude 42.7371538 degrees, longitude -73.6692229 degrees. The world
  rectangle's corner is **not** the origin.
- Yaw: radians, zero east, positive counterclockwise toward north.
- vx/vy and filtered accel_x/accel_y: body-frame m/s and m/s².
- Yaw rate: rad/s; rear-wheel peripheral speed: **m/s**, not RPM or rad/s.
- Measured and commanded steering: road-wheel radians, not steering-wheel angle.
- Measured and commanded torque: rear-wheel Nm, not throttle or engine torque.
- dt: seconds; trained at 0.02 s (50 Hz).

The artifact embeds the curve, x/y/yaw-to-s/e/dphi conversion, derived speed
and sideslip, input normalization, NN weights, rate integration, and clipping.
The NN predicts rate changes, bounded by 0.6 rad/s steering and 3000 Nm/s torque.
These integrate from **previous commanded controls**, not measured controls.
Outputs are absolute road-wheel steering in [-0.4, 0.4] rad and rear-wheel
torque in [-400, 1000] Nm. Both measured controls and previous commands are
network inputs; they coincided in the actuator-dynamics-disabled training.
No previous-s or recurrent memory is needed. Caller handles lap completion,
state estimation, actuation, and safety; none is included in the export.

## Closed-loop prediction: policy_unroll

```python
from policy_unroll import policy_unroll, STATE_FIELDS

# Same measured[11] and previous[2] inputs as policy.jax.
states, commands = policy_unroll(measured, previous, dt=0.02, N=100)
# states.shape == (101, 14), commands.shape == (100, 2)
s, e, dphi = states[:, 11], states[:, 12], states[:, 13]
```

`policy_unroll.jax` bundles the policy, exptanh dynamics checkpoint 5800, track,
and all conversions. The Python loader needs only this artifact and JAX;
`policy.jax` is not separately required for unrolling. N is a Python/static
integer; changing N can cause compilation. Warm up the desired horizon first.
The low-level serialized function accepts `(measured[11], previous[2], dts[N])`;
the wrapper supplies N copies of dt. N=0 returns only the initial state.

State columns, in order:

```text
x, y, yaw, vx, vy, yaw_rate, rear_wheel_speed, accel_x, accel_y,
measured_steer, measured_torque, s, e, dphi
```

All units and frames match the policy inputs above. Acceleration channels are
the model's filtered `fcog_x/fcog_y` states, not an independent IMU simulation.
`dphi` is course-heading error (body yaw + sideslip - reference heading).
State 0 is the supplied measurement plus its projected s/e/dphi. Command k
is computed from state k, held over dt, and produces state k+1. Predicted
measured controls equal that applied command because actuator lag is disabled.
Initial measured controls and previous commands may differ; both are retained
for the first policy call. Speed and beta can be derived from vx/vy.

The integrator is the same forward-Euler/Frenet update as training, with **five
equal dt/5 substeps**. Training randomized the substep partition; this export
is deterministic and does not reproduce that randomness. Trained control dt
is 20 ms; arbitrary positive dt is accepted but large dt can be inaccurate or
unstable. No stopping/reset occurs at e-bounds, lap completion, or timeout;
s remains unwrapped from its initial one-lap projection. Long predictions,
out-of-distribution states, and model-to-real error require caller judgment.
There is no collision checking or safety supervisor.

Run standalone regression tests with:

```bash
JAX_PLATFORMS=cpu python -m unittest test_policy_unroll
```

## Provenance and limits

Run `colony_exp_limited_ft`, privileged actor, best checkpoint `step_000170000384`.
Trained against `can_am_x3_r322_exptanh_planar____`, checkpoint 5800, from
model repository commit `35483d0`. Fine-tuned from `colony_exp` checkpoint
225050624 for 240.12M additional frames in about 79 seconds on RTX 5080;
the selected actor is from 170.00M fine-tuning frames, after spawn annealing.
The preceding from-scratch limited-torque attempt failed; it is not published.
Earlier policy versions remain available in Git history.

Track: existing `colony_drive.py` slide_sweep, clockwise, 20 m straights,
8 m turn radius, 3 m curvature transitions, ENU bounding-box center (12, -26),
one 96.26548 m lap. Full ±2.25 m corridor has 1.65 m minimum sampled survey
clearance and clears the surveyed pole keepout; this does not certify full
vehicle footprint clearance. Episodes terminate on e-bound exits, completion,
or timeout, not collisions. No actuator lag or parameter randomization.

Reward: forward progress minus lateral/heading error, overspeed, and normalized
control-rate costs; a soft 6 m/s speed target, not a hard limiter. Added lateral
acceleration cost is `dt * 30 * max(abs(ay) - 7, 0)**2`, using filtered fcog_y
in m/s². This is a soft penalty, not an acceleration clamp. Speed and s
spawns anneal during 15--55% of training from vx=[0.08,4] m/s, s=[0,10] m to
vx=[0.08,0.15] m/s, s=[0,0.5] m. Initial e is ±0.7875 m, sideslip zero.

Held-out simulation seed 20260902: 128/128 completions, mean lap 20.54 s,
RMS e 0.155 m, peak speed 6.177 m/s, peak |ay| 4.201 m/s² with no samples above
7 m/s²; observed commands remain in [-400,1000] Nm.
**No real-car transfer test has been done.**
This folder contains inference only, not an optimizer/resumable PPO checkpoint.
