"""Self-contained closed-loop prediction; no vehicle or training repo imports."""
from functools import lru_cache
from pathlib import Path
import operator

import jax
import jax.numpy as jnp
from jax import export

STATE_FIELDS = (
    'x_enu_m', 'y_enu_m', 'yaw_enu_rad', 'vx_body_ms', 'vy_body_ms',
    'yaw_rate_rads', 'rear_wheel_speed_ms', 'accel_x_ms2', 'accel_y_ms2',
    'measured_steer_rad', 'measured_rear_torque_nm', 's_m', 'e_m', 'dphi_rad',
)


@lru_cache(maxsize=1)
def _load():
    data = Path(__file__).with_name('policy_unroll.jax').read_bytes()
    return jax.jit(export.deserialize(data).call)


def policy_unroll(measured, previous_command, dt, N):
    """Return states[N+1,14], commands[N,2]; dt seconds, N a static integer.

    Includes the initial state; no episode termination or reset. Five equal
    Euler substeps per tick. Supply finite SI inputs, positive dt, and bounded
    previous commands. This is a prediction, not a safety controller.
    """
    N = operator.index(N)
    if N < 0:
        raise ValueError('N must be nonnegative')
    measured = jnp.asarray(measured, jnp.float32)
    previous_command = jnp.asarray(previous_command, jnp.float32)
    dt = jnp.asarray(dt, jnp.float32)
    if measured.shape != (11,) or previous_command.shape != (2,) or dt.shape != ():
        raise ValueError('Expected measured[11], previous_command[2], scalar dt')
    # Shape-polymorphic exported scan has N>=1. For N=0, return only its
    # reconstructed initial record and an empty command array.
    states, commands = _load()(measured, previous_command,
        jnp.full((max(N, 1),), dt, jnp.float32))
    return states[:N+1], commands[:N]
