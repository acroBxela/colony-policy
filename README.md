# Colony Can-Am policy (`colony_exp`)

Self-contained **CPU JAX/StableHLO inference artifact**, not a dynamics model,
standalone executable, or real-car safety controller. No imports from the RL
project or the vehicle-model repository are needed at inference time.
This standalone repository contains only the exported policy and its loader,
not the XAL source code, dynamics checkpoints, or training data.

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
torque in [-600, 2000] Nm. Both measured controls and previous commands are
network inputs; they coincided in the actuator-dynamics-disabled training.
No previous-s or recurrent memory is needed. Caller handles lap completion,
state estimation, actuation, and safety; none is included in the export.

## Provenance and limits

Run `colony_exp`, privileged actor, best checkpoint `step_000225050624`.
Trained against `can_am_x3_r322_exptanh_planar____`, checkpoint 5800, from
model repository commit `35483d0`. The run trained 240.12M frames in about
77.3 seconds on RTX 5080; the selected actor is from 225.05M frames.

Track: existing `colony_drive.py` slide_sweep, clockwise, 20 m straights,
8 m turn radius, 3 m curvature transitions, ENU bounding-box center (12, -26),
one 96.26548 m lap. Full ±2.25 m corridor has 1.65 m minimum sampled survey
clearance and clears the surveyed pole keepout; this does not certify full
vehicle footprint clearance. Episodes terminate on e-bound exits, completion,
or timeout, not collisions. No actuator lag or parameter randomization.

Reward: forward progress minus lateral/heading error, overspeed, and normalized
control-rate costs; a soft 6 m/s speed target, not a hard limiter. Speed and s
spawns anneal during 15--55% of training from vx=[0.08,4] m/s, s=[0,10] m to
vx=[0.08,0.15] m/s, s=[0,0.5] m. Initial e is ±0.7875 m, sideslip zero.

Held-out simulation seed 20260902: 128/128 completions, mean lap 20.23 s,
RMS e 0.164 m, peak speed 6.213 m/s. **No real-car transfer test has been done.**
This folder contains inference only, not an optimizer/resumable PPO checkpoint.
