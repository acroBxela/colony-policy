"""Standalone export checks: no training repository or vehicle model imports."""
import unittest
import numpy as np
import jax.numpy as jnp
from load_policy import load_policy
from policy_unroll import policy_unroll


class UnrollTest(unittest.TestCase):
    def setUp(self):
        self.m = jnp.array([3.9531841, -36, 1.5707963, .1, 0, 0, .1,
                           0, 0, 0, 250], jnp.float32)
        self.p = jnp.array([0, 250], jnp.float32)

    def test_shapes_initial_and_first_control(self):
        for n in (0, 1, 10):
            s, a = policy_unroll(self.m, self.p, .02, n)
            self.assertEqual(s.shape, (n+1, 14))
            self.assertEqual(a.shape, (n, 2))
            np.testing.assert_array_equal(s[0, :11], self.m)
            if n:
                np.testing.assert_allclose(a[0], load_policy()(self.m, self.p,
                    jnp.float32(.02)), atol=1e-5)
                np.testing.assert_array_equal(s[1:, 9:11], a)

    def test_distinct_measured_and_commanded_controls(self):
        m = self.m.at[9].set(.02).at[10].set(100)
        s, a = policy_unroll(m, self.p, .02, 1)
        np.testing.assert_array_equal(s[0, :11], m)
        np.testing.assert_allclose(a[0], load_policy()(m, self.p,
            jnp.float32(.02)), atol=1e-5)

    def test_complete_horizon_and_unwrapped_progress(self):
        s, a = policy_unroll(self.m, self.p, .02, 1100)
        self.assertTrue(np.isfinite(s).all())
        self.assertGreater(float(s[-1, 11]), 96.26548)
        self.assertLess(float(np.abs(s[:, 12]).max()), 2.25)
        self.assertTrue(np.all(np.abs(a[:, 0]) <= .400001))
        self.assertTrue(np.all((a[:, 1] >= -600.001) & (a[:, 1] <= 2000.001)))

    def test_input_shapes_and_negative_horizon(self):
        with self.assertRaises(ValueError):
            policy_unroll(self.m, self.p, .02, -1)
        with self.assertRaises(ValueError):
            policy_unroll(self.m[:10], self.p, .02, 1)


if __name__ == '__main__':
    unittest.main()
