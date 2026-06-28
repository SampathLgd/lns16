"""
Unit tests for LNS16 library
Run: python -m pytest tests/test_lns16.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest
from lns_lib.lns16 import (
    _float_to_lns16_np, _lns16_to_float_np,
    lns_logmul, lns_logadd, lns_dot_product, LNS16
)


# ─── Encode / Decode ──────────────────────────────────────────────────────────

class TestEncodeDecoder:
    VALUES = [1.0, -1.0, 0.5, -0.5, 2.0, -4.0, 0.125, 1000.0, -0.001, 0.0]

    def test_round_trip(self):
        for v in self.VALUES:
            arr  = np.array([v], dtype=np.float64)
            bits = _float_to_lns16_np(arr)
            recon = _lns16_to_float_np(bits)[0]
            if v == 0.0:
                assert recon == 0.0
            else:
                rel_err = abs(recon - v) / abs(v)
                assert rel_err < 0.005, f"v={v}, recon={recon}, rel_err={rel_err:.6f}"

    def test_zero(self):
        bits = _float_to_lns16_np(np.array([0.0]))
        assert _lns16_to_float_np(bits)[0] == 0.0

    def test_sign_preserved(self):
        for v in [-1.0, -3.14, -100.0]:
            bits = _float_to_lns16_np(np.array([v]))
            recon = _lns16_to_float_np(bits)[0]
            assert recon < 0, f"Expected negative for {v}, got {recon}"

    def test_batch(self):
        arr  = np.linspace(-10, 10, 100)
        bits = _float_to_lns16_np(arr)
        recon = _lns16_to_float_np(bits)
        # Allow for zero entries and relative error
        nz = arr != 0
        rel = np.abs((recon[nz] - arr[nz]) / arr[nz])
        assert rel.max() < 0.01


# ─── logmul ───────────────────────────────────────────────────────────────────

class TestLogMul:
    def _mul(self, a, b):
        ab = _float_to_lns16_np(np.array([a], dtype=np.float64))
        bb = _float_to_lns16_np(np.array([b], dtype=np.float64))
        return _lns16_to_float_np(lns_logmul(ab, bb))[0]

    def test_positive(self):
        assert abs(self._mul(2.0, 3.0) - 6.0) / 6.0 < 0.005

    def test_negative(self):
        assert self._mul(-2.0, 3.0) < 0
        assert abs(self._mul(-2.0, 3.0) + 6.0) / 6.0 < 0.005

    def test_neg_neg(self):
        assert self._mul(-2.0, -3.0) > 0

    def test_zero(self):
        assert self._mul(0.0, 5.0) == 0.0
        assert self._mul(5.0, 0.0) == 0.0

    def test_identity(self):
        # x * 1 ≈ x
        for v in [1.5, -3.0, 0.25]:
            assert abs(self._mul(v, 1.0) - v) / abs(v) < 0.005


# ─── logadd ───────────────────────────────────────────────────────────────────

class TestLogAdd:
    def _add(self, a, b):
        ab = _float_to_lns16_np(np.array([a], dtype=np.float64))
        bb = _float_to_lns16_np(np.array([b], dtype=np.float64))
        return _lns16_to_float_np(lns_logadd(ab, bb))[0]

    def test_positive(self):
        assert abs(self._add(2.0, 3.0) - 5.0) / 5.0 < 0.005

    def test_negative(self):
        assert abs(self._add(-1.0, -2.0) + 3.0) / 3.0 < 0.005

    def test_zero(self):
        assert abs(self._add(0.0, 4.0) - 4.0) / 4.0 < 0.005

    def test_cancellation(self):
        result = self._add(3.0, -3.0)
        assert abs(result) < 0.1   # should be near zero


# ─── dot product ──────────────────────────────────────────────────────────────

class TestDotProduct:
    def _dot(self, a, b):
        ab = _float_to_lns16_np(np.array(a, dtype=np.float64))
        bb = _float_to_lns16_np(np.array(b, dtype=np.float64))
        return _lns16_to_float_np(lns_dot_product(ab, bb))[0]

    def test_simple(self):
        assert abs(self._dot([1, 2, 3], [4, 5, 6]) - 32.0) / 32.0 < 0.01

    def test_unit_vectors(self):
        assert abs(self._dot([1, 0, 0], [1, 0, 0]) - 1.0) < 0.01

    def test_orthogonal(self):
        result = self._dot([1, 0], [0, 1])
        assert abs(result) < 0.05


# ─── LNS16 class ──────────────────────────────────────────────────────────────

class TestLNS16Class:
    def test_mul(self):
        a = LNS16(torch.tensor([2.0, 3.0]))
        b = LNS16(torch.tensor([4.0, 5.0]))
        c = (a * b).tensor.numpy()
        np.testing.assert_allclose(c, [8.0, 15.0], rtol=0.01)

    def test_add(self):
        a = LNS16(torch.tensor([1.0, 2.0]))
        b = LNS16(torch.tensor([3.0, 4.0]))
        c = (a + b).tensor.numpy()
        np.testing.assert_allclose(c, [4.0, 6.0], rtol=0.01)

    def test_quantise_tensor(self):
        t = torch.tensor([0.5, 1.0, -2.0, 0.0])
        q = LNS16.quantise_tensor(t)
        # Should be close to original
        np.testing.assert_allclose(q.numpy(), t.numpy(), rtol=0.01, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
