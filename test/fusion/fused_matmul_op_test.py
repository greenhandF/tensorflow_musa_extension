# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the MusaFusedMatMul operator."""

import os
import unittest

os.environ.setdefault("MUSA_ENABLE_TF32", "0")

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import op_def_registry

from musa_test_utils import MUSATestCase, load_musa_ops


def is_tf32_enabled():
    return int(os.environ.get("MUSA_ENABLE_TF32", "0")) != 0


def float32_tolerance(default_rtol=1e-5, default_atol=1e-5):
    return (1e-2, 1e-2) if is_tf32_enabled() else (default_rtol, default_atol)


class FusedMatMulOpTest(MUSATestCase):
    """Functional tests for MusaFusedMatMul."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if op_def_registry.get("MusaFusedMatMul") is None:
            raise unittest.SkipTest(
                "MusaFusedMatMul is not registered. Rebuild plugin first."
            )

        musa_ops = load_musa_ops()
        for wrapper_name in ("musa_fused_mat_mul", "musa_fused_matmul"):
            if hasattr(musa_ops, wrapper_name):
                cls._fused_matmul = staticmethod(getattr(musa_ops, wrapper_name))
                break
        else:
            raise unittest.SkipTest("MusaFusedMatMul Python wrapper is not available.")

    def _run_fused_matmul(
        self,
        a_np,
        b_np,
        bias_np,
        dtype=tf.float32,
        fused_ops=("BiasAdd",),
        transpose_a=False,
        transpose_b=False,
    ):
        with tf.device("/device:MUSA:0"):
            a = tf.constant(a_np, dtype=dtype)
            b = tf.constant(b_np, dtype=dtype)
            bias = tf.constant(bias_np, dtype=dtype)
            out = self._fused_matmul(
                a=a,
                b=b,
                bias=bias,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                fused_ops=list(fused_ops),
            )
            return tf.cast(out, tf.float32).numpy()

    def _run_reference(
        self,
        a_np,
        b_np,
        bias_np,
        dtype=tf.float32,
        fused_ops=("BiasAdd",),
        transpose_a=False,
        transpose_b=False,
    ):
        with tf.device("/CPU:0"):
            a = tf.constant(a_np, dtype=dtype)
            b = tf.constant(b_np, dtype=dtype)
            bias = tf.constant(bias_np, dtype=dtype)
            out = tf.matmul(
                a, b, transpose_a=transpose_a, transpose_b=transpose_b
            )
            out = tf.nn.bias_add(out, bias)
            if tuple(fused_ops) == ("BiasAdd", "Relu"):
                out = tf.nn.relu(out)
            return tf.cast(out, tf.float32).numpy()

    def _assert_fused_matches_reference(
        self,
        a_np,
        b_np,
        bias_np,
        dtype=tf.float32,
        fused_ops=("BiasAdd",),
        transpose_a=False,
        transpose_b=False,
        rtol=None,
        atol=None,
    ):
        expected = self._run_reference(
            a_np,
            b_np,
            bias_np,
            dtype=dtype,
            fused_ops=fused_ops,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )
        actual = self._run_fused_matmul(
            a_np,
            b_np,
            bias_np,
            dtype=dtype,
            fused_ops=fused_ops,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

        if rtol is None or atol is None:
            if dtype == tf.float32:
                rtol, atol = float32_tolerance()
            elif dtype == tf.float16:
                rtol, atol = 1e-2, 1e-2
            elif dtype == tf.bfloat16:
                rtol, atol = 2e-2, 2e-2
            else:
                rtol, atol = 1e-5, 1e-5

        self.assertAllClose(actual, expected, rtol=rtol, atol=atol)

    def test_bias_add_float32(self):
        rng = np.random.default_rng(123)
        a_np = rng.standard_normal((4, 8)).astype(np.float32)
        b_np = rng.standard_normal((8, 6)).astype(np.float32)
        bias_np = rng.standard_normal((6,)).astype(np.float32)

        self._assert_fused_matches_reference(a_np, b_np, bias_np)

    def test_bias_add_relu_float32(self):
        rng = np.random.default_rng(456)
        a_np = rng.standard_normal((5, 7)).astype(np.float32)
        b_np = rng.standard_normal((7, 9)).astype(np.float32)
        bias_np = rng.standard_normal((9,)).astype(np.float32)

        self._assert_fused_matches_reference(
            a_np, b_np, bias_np, fused_ops=("BiasAdd", "Relu")
        )

    def test_transpose_a_and_b(self):
        rng = np.random.default_rng(789)
        a_np = rng.standard_normal((6, 4)).astype(np.float32)
        b_np = rng.standard_normal((5, 6)).astype(np.float32)
        bias_np = rng.standard_normal((5,)).astype(np.float32)

        self._assert_fused_matches_reference(
            a_np,
            b_np,
            bias_np,
            fused_ops=("BiasAdd", "Relu"),
            transpose_a=True,
            transpose_b=True,
        )

    def test_rank4_batch_broadcast_bias_add_relu(self):
        rng = np.random.default_rng(321)
        a_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
        b_np = rng.standard_normal((1, 1, 5, 6)).astype(np.float32)
        bias_np = rng.standard_normal((6,)).astype(np.float32)

        self._assert_fused_matches_reference(
            a_np, b_np, bias_np, fused_ops=("BiasAdd", "Relu")
        )

    def test_dtypes_bias_add(self):
        rng = np.random.default_rng(654)
        a_np = rng.standard_normal((3, 5)).astype(np.float32)
        b_np = rng.standard_normal((5, 4)).astype(np.float32)
        bias_np = rng.standard_normal((4,)).astype(np.float32)

        for dtype in (tf.float32, tf.float16, tf.bfloat16):
            self._assert_fused_matches_reference(
                a_np, b_np, bias_np, dtype=dtype, fused_ops=("BiasAdd",)
            )

    def test_empty_inner_dim_bias_add_relu(self):
        a_np = np.empty((3, 0), dtype=np.float32)
        b_np = np.empty((0, 4), dtype=np.float32)
        bias_np = np.array([-1.0, 0.5, 2.0, -0.25], dtype=np.float32)

        self._assert_fused_matches_reference(
            a_np,
            b_np,
            bias_np,
            fused_ops=("BiasAdd", "Relu"),
            rtol=0,
            atol=0,
        )

    def test_invalid_dim_mismatch(self):
        rng = np.random.default_rng(987)
        a_np = rng.standard_normal((2, 7)).astype(np.float32)
        b_np = rng.standard_normal((8, 4)).astype(np.float32)
        bias_np = rng.standard_normal((4,)).astype(np.float32)

        with self.assertRaises(Exception):
            _ = self._run_fused_matmul(a_np, b_np, bias_np)


if __name__ == "__main__":
    tf.test.main()
