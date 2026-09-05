"""Load the self-contained CPU policy; running this file only smoke-tests inference."""
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import export


def load_policy():
    artifact = Path(__file__).with_name("policy.jax")
    return jax.jit(export.deserialize(artifact.read_bytes()).call)


if __name__ == "__main__":
    policy = load_policy()
    # Synthetic near-rest start, NOT a vehicle interface or deployment command.
    measured = jnp.array([
        3.9531841, -36.0, 1.5707963, 0.1, 0.0, 0.0, 0.1,
        0.0, 0.0, 0.0, 250.0,
    ], dtype=jnp.float32)
    previous = jnp.array([0.0, 250.0], dtype=jnp.float32)
    command = policy(measured, previous, jnp.float32(0.02))
    command.block_until_ready()  # First call compiles; warm up before a control loop.
    assert command.shape == (2,) and bool(jnp.all(jnp.isfinite(command)))
    assert abs(float(command[0])) <= 0.4
    assert -400.0 <= float(command[1]) <= 1000.0
    assert bool(jnp.all(jnp.abs(command - previous) <= jnp.array([0.012001, 60.001])))
    print("Smoke test only: [steering radians, rear-wheel torque Nm] =", command.tolist())
