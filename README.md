# Colony policy collection

Self-contained JAX policy and closed-loop prediction exports for the Can-Am.
Each named directory bundles a matched policy, dynamics unroll, metadata,
loaders, tests, and usage notes. Do not mix artifacts from different directories.
No XAL installation is needed. These are simulation-tested research policies,
**not real-car validated safety controllers**.

## Available policies

| Policy | Torque bounds (Nm) | Lateral acceleration cost | Simulation result |
| --- | --- | --- | --- |
| [colony_exp_limited_ft](policies/colony_exp_limited_ft/README.md) — latest | −400 to 1000 | Squared excess above \|ay\| = 7 m/s², weight 30 | 128/128 completed; 20.54 s mean lap |
| [colony_exp](policies/colony_exp/README.md) — previous | −600 to 2000 | None | 128/128 completed; 20.23 s mean lap |

Both use `can_am_x3_r322_exptanh_planar____`, dynamics checkpoint 5800,
the same Colony track, and the same input/output interface. Results are on
each policy's own deterministic simulation evaluation; they are not guarantees.
The current limited-torque policy is checkpoint `step_000170000384`, fine-tuned
from the previous policy. Earlier versions remain separately usable.

## Quick start

```bash
git clone https://github.com/acroBxela/colony-policy.git
cd colony-policy/policies/colony_exp_limited_ft
python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu python load_policy.py
JAX_PLATFORMS=cpu python -m unittest test_policy_unroll
```

From the repository root, the loaders also support named imports:

```python
from policies.colony_exp_limited_ft.load_policy import load_policy
from policies.colony_exp_limited_ft.policy_unroll import policy_unroll

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
