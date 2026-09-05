# Colony Can-Am policy: colony_fiala

Self-contained CPU JAX/StableHLO policy and closed-loop model prediction.
Selected checkpoint: `step_000210108416`. Trained from scratch, not fine-tuned.
No real-vehicle testing or safety certification has been performed.

## Load and run

Tested with Python 3.11 and JAX 0.7.2. Install `requirements.txt` in a separate
environment. No RL repository or XAL imports are required at inference time.

```python
import jax.numpy as jnp
from load_policy import load_policy
from policy_unroll import policy_unroll, STATE_FIELDS

policy = load_policy()
measured = jnp.array([
    3.9531841, -36., 1.5707963, .1, 0., 0., .1, 0., 0., 0., 250.
], dtype=jnp.float32)
previous = jnp.array([0., 250.], dtype=jnp.float32)
command = policy(measured, previous, jnp.float32(.01))  # 100 Hz
states, commands = policy_unroll(measured, previous, dt=.01, N=100)
# states: (101,14); commands: (100,2). Initial state is included.
previous = command  # Caller retains previous command for the next live tick.
```

Measured input order (physical SI units; normalization is embedded):
`x, y, yaw, vx, vy, yaw_rate, rear_wheel_speed, accel_x, accel_y, measured_steer, measured_torque`.
Previous command order and output order: `steer_rad, rear_torque_Nm`.
Unroll states append `s, e, dphi` to the eleven measured fields above.
The low-level unroll artifact takes `(measured[11], previous[2], dts[N])`.

Coordinates are Colony's surveyed local ENU: x east, y north, yaw zero east,
positive counterclockwise. Survey origin: latitude 42.7371538, longitude
-73.6692229 degrees; the rectangle corner is not the origin.
vx/vy and filtered accel_x/accel_y are body-frame quantities.
Rear wheel speed is peripheral **m/s**, not RPM. Steering is road-wheel radians;
torque is rear-wheel Nm. dphi is course-heading error (yaw + beta - track heading).

The policy embeds track projection, derived features, normalization, NN weights,
rate integration and clipping. Outputs are absolute commands, bounded to
steer +/-0.4 rad and torque [-400,1000] Nm. Rates are bounded to 0.6 rad/s and
3000 Nm/s, integrated from previous commanded controls, not measured controls.
Both measured and previous commanded controls are inputs. No recurrent memory
or previous s is needed by the policy. Warm up and synchronize before use;
changing unroll N may compile. Supply finite inputs and positive dt.

## Training and evaluation

- Model: `can_am_x3_r322_fiala_planar_vnew2`, checkpoint 1600, model source
  commit `35483d0`. No actuator lag. Independent uniform +/-3% episode perturbations of front Cf
  and front/rear longitudinal and lateral friction coefficients; rear Cr unchanged.
  Exported unroll and nominal trajectory use unperturbed identified parameters.
- 240,123,904 training frames in approximately 86 seconds on RTX 5080;
  selected actor at 210,108,416 frames.
- Control frequency sampled uniformly in **[50,130] Hz per episode**, then held.
  Each control interval is partitioned into five randomized physics substeps.
- Starting s uniform over the full 96.2654824574 m lap, throughout training.
  Starting e uniform +/-0.7875 m, initial sideslip and yaw rate zero.
  vx anneals from [0.08,4] to [0.08,0.15] m/s over 15–55% of training.
- A full lap is required from each spawn; e-bound exits terminate training episodes.
- Reward: forward progress, minus dt times lateral error squared (weight 12),
  heading cost (8), overspeed above 6 m/s squared (30), steering/torque normalized
  rate squared (0.1 each), and max(abs(filtered ay)-7,0) squared (30).
  Finish bonus 20; failure penalty 20. Speed/ay limits are soft costs, not clamps.
- Independent 128-seed evaluation: 128/128 completed; mean lap 18.4846 s,
  RMS e 0.1516 m, peak |e| 0.7897 m, peak |ay| 4.4250 m/s².

[Trajectories](plots/trajectory_rollouts.png) ·
[States and controls](plots/states_and_controls_vs_s.png)

## Nominal trajectory

`nominal_trajectory.zip` and `nominal_trajectory/` contain `traj.csv`,
`surface_track.npz`, `meta.yaml`, `stats.json`, and native trajopt plots.
This is a closed-loop rollout of this exact policy at **50 Hz**, starting at
s=e=beta=0, vx=0.1 m/s, previous torque=250 Nm. Startup is retained; only the
final crossing is interpolated to exact lap-end s. It is not an optimized path.
Nominal result: 939 samples, 18.7570 s lap, peak |e| 0.2532 m.
Surface reconstruction maximum x/y error: 0.0000072 m.

```python
from drive_trajopt.traj import load_artifact
trajectory = load_artifact('nominal_trajectory')
```

Plot with `python -m drive_trajopt.traj nominal_trajectory` in an environment
with drive_trajopt installed. Use active track coordinates s_track/e_track/
dphi_track; surface_track is the original corridor, not a fitted reference path.
Row k command applies to the next interval; terminal row holds the last command.

## Prediction limitations and checks

Unroll embeds the policy and frozen dynamics, using the training Frenet Euler
integrator with five **equal** dt/5 substeps. It is deterministic, with no resets
or termination at lap ends or bounds; s is unwrapped after initial projection.
It is **not the Cartesian diagnostic integrator**: far outside the track,
Frenet coordinates can become singular and reconstructed x/y unreliable.
Do not treat those off-track predictions as physical trajectories.
Policy x/y projection itself is embedded and stateless.

There is no vehicle interface, collision checking, state estimator, safety
supervisor, or resumable PPO optimizer in this package. Inference/model
prediction only. Load trusted artifacts and validate on the target hardware.

```bash
JAX_PLATFORMS=cpu python load_policy.py
JAX_PLATFORMS=cpu python -m unittest test_policy_unroll
```
