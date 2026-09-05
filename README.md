# Colony policy collection

Self-contained JAX policy and closed-loop prediction exports for the Can-Am.
Each named directory bundles a matched policy, dynamics unroll, metadata,
loaders, tests, and usage notes. Do not mix artifacts from different directories.
No XAL installation is needed. These are simulation-tested research policies,
**not real-car validated safety controllers**.

## Available policies

| Policy | Torque bounds (Nm) | Lateral acceleration cost | Simulation result |
| --- | --- | --- | --- |
| [colony_fiala_e20_h25](policies/colony_fiala_e20_h25/README.md) — latest | −400 to 1000 | Squared excess above \|ay\| = 7 m/s², weight 30 | Wider spawns: 128/128 completed; 22.11 s mean lap; RMS e 0.380 m |
| [colony_fiala](policies/colony_fiala/README.md) | −400 to 1000 | Squared excess above \|ay\| = 7 m/s², weight 30 | 128/128 completed; 18.48 s mean lap; RMS e 0.152 m |
| [colony_hz_scratch](policies/colony_hz_scratch/README.md) | −400 to 1000 | Squared excess above \|ay\| = 7 m/s², weight 30 | 128/128 completed; 20.69 s mean lap; RMS e 0.182 m |
| [colony_exp_limited_ft](policies/colony_exp_limited_ft/README.md) | −400 to 1000 | Squared excess above \|ay\| = 7 m/s², weight 30 | 128/128 completed; 20.54 s mean lap |
| [colony_exp](policies/colony_exp/README.md) — previous | −600 to 2000 | None | 128/128 completed; 20.23 s mean lap |

The `colony_fiala` policies use `can_am_x3_r322_fiala_planar_vnew2`, checkpoint 1600,
with independent +/-3% front Cf and front/rear longitudinal/lateral friction
randomization during training. Its exported unroll uses nominal parameters.
Older policies use `can_am_x3_r322_exptanh_planar____`, checkpoint 5800.
All share the same Colony track and input/output interface. Results are on
each policy's own deterministic simulation evaluation; they are not guarantees.
The latest policy was fine-tuned from a working recovery policy, with uniformly
randomized 50–130 Hz control frequency per episode, full-lap randomized starts,
e in +/-2 m, heading in +/-25 degrees, vx in [0.08,0.15] m/s, lateral-error
weight 12, and failure penalty 100. Its selected checkpoint is
`step_000010092544` (10M additional frames), the successful early checkpoint
viewed in drive_agent; later collapsed checkpoints are not exported. Earlier versions
remain separately usable. Evaluation spawn/timing distributions differ across
policies, as do the dynamics, so lap times are not a controlled head-to-head comparison.

## Quick start

```bash
git clone https://github.com/acroBxela/colony-policy.git
cd colony-policy/policies/colony_fiala_e20_h25
python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu python load_policy.py
JAX_PLATFORMS=cpu python -m unittest test_policy_unroll
```

From the repository root, the loaders also support named imports:

```python
from policies.colony_fiala_e20_h25.load_policy import load_policy
from policies.colony_fiala_e20_h25.policy_unroll import policy_unroll

policy = load_policy()
command = policy(measured, previous_command, dt)
states, commands = policy_unroll(measured, previous_command, dt=0.02, N=100)
```

Read the chosen policy's README for SI units, ENU coordinates, warmup, output
columns, and integration limitations. Use float32 inputs. `policy_unroll` uses
five equal physics substeps per control tick, includes the initial state, and
does not terminate or reset at track bounds or lap completion.

## Layout and future additions

```text
policies/<run_name>/
  policy.jax                 # Single-step policy
  policy_unroll.jax          # Matched policy + dynamics + map
  load_policy.py
  policy_unroll.py
  metadata.json
  unroll_metadata.json
  requirements.txt
  test_policy_unroll.py
  README.md
  plots/                     # Optional recorded evaluation images
```

Add future policies under new run names and update this index. Keep each
policy/unroll pair together; do not silently replace an existing version with
different weights. The former root-level artifacts are now under
`policies/colony_exp/`.
