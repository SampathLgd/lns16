"""
LNS16 Core Implementation
=========================
16-bit Logarithmic Number System:
  Bit layout: [sign(1) | integer_log(6) | frac_log(9)]
  Represents x = (-1)^sign * 2^(integer + frac/512)
"""

import numpy as np
import torch

INT_BITS      = 6
FRAC_BITS     = 9
FRAC_SCALE    = 2 ** FRAC_BITS        # 512
INT_BIAS      = 2 ** (INT_BITS - 1)   # 32
ZERO_SENTINEL = 0xFFFF


# ─── NumPy encode/decode (used only for unit tests / small arrays) ────────────

def _float_to_lns16_np(x):
    x = np.asarray(x, dtype=np.float64)
    zero_mask = (x == 0.0)
    sign_mask = (x < 0)
    safe      = np.where(zero_mask, 1.0, np.abs(x))
    log2      = np.log2(safe)
    fixed     = np.round(log2 * FRAC_SCALE).astype(np.int32)
    ip        = np.clip(fixed >> FRAC_BITS, -INT_BIAS, INT_BIAS - 1)
    fp        = (fixed & (FRAC_SCALE - 1)).astype(np.uint16)
    ib        = (ip + INT_BIAS).astype(np.uint16)
    s         = sign_mask.astype(np.uint16)
    packed    = (s << 15) | (ib << FRAC_BITS) | fp
    return np.where(zero_mask, np.uint16(ZERO_SENTINEL), packed).astype(np.uint16)


def _lns16_to_float_np(bits):
    bits = np.asarray(bits, dtype=np.uint16)
    zero = (bits == ZERO_SENTINEL)
    s    = ((bits >> 15) & 1).astype(np.float64)
    ib   = ((bits >> FRAC_BITS) & ((1 << INT_BITS) - 1)).astype(np.int32)
    fr   = (bits & (FRAC_SCALE - 1)).astype(np.float64)
    log2 = (ib - INT_BIAS).astype(np.float64) + fr / FRAC_SCALE
    mag  = np.exp2(log2)
    out  = np.where(s == 0, mag, -mag)
    return np.where(zero, 0.0, out)


# ─── Fast PyTorch quantise (no NumPy, stays on device) ───────────────────────

def _quantise_torch(t: torch.Tensor) -> torch.Tensor:
    """
    Round-trip a float32 tensor through LNS16 quantisation using pure PyTorch.
    This is the fast path used during model inference.
    """
    orig_dtype = t.dtype
    x = t.float()

    zero_mask = (x == 0.0)
    sign_mask = (x < 0.0)
    safe      = x.abs().clamp(min=1e-38)

    log2_val  = torch.log2(safe)                          # float32 log2
    fixed     = torch.round(log2_val * FRAC_SCALE).to(torch.int32)

    ip        = fixed >> FRAC_BITS                        # integer part
    fp        = (fixed & (FRAC_SCALE - 1)).float()        # fractional part

    ip_clamped = ip.clamp(-INT_BIAS, INT_BIAS - 1).float()

    # Reconstruct float from quantised log
    log2_recon = ip_clamped + fp / FRAC_SCALE
    magnitude  = torch.exp2(log2_recon)

    # Apply sign
    result = torch.where(sign_mask, -magnitude, magnitude)

    # Zeros stay zero
    result = torch.where(zero_mask, torch.zeros_like(result), result)

    return result.to(orig_dtype)


# ─── NumPy arithmetic (logmul / logadd / dot) ────────────────────────────────

def lns_logmul(a_bits, b_bits):
    a_bits = np.asarray(a_bits, dtype=np.uint16)
    b_bits = np.asarray(b_bits, dtype=np.uint16)
    zero_out = (a_bits == ZERO_SENTINEL) | (b_bits == ZERO_SENTINEL)

    def lf(x):
        ib = ((x >> FRAC_BITS) & ((1 << INT_BITS) - 1)).astype(np.int32)
        fr = (x & (FRAC_SCALE - 1)).astype(np.int32)
        return ((ib - INT_BIAS) << FRAC_BITS) | fr

    log_c  = lf(a_bits) + lf(b_bits)
    sign_c = ((a_bits >> 15) ^ (b_bits >> 15)).astype(np.uint16)
    ic     = np.clip(log_c >> FRAC_BITS, -INT_BIAS, INT_BIAS - 1)
    fc     = (log_c & (FRAC_SCALE - 1)).astype(np.uint16)
    packed = (sign_c << 15) | ((ic + INT_BIAS).astype(np.uint16) << FRAC_BITS) | fc
    return np.where(zero_out, np.uint16(ZERO_SENTINEL), packed).astype(np.uint16)


def lns_logadd(a_bits, b_bits):
    fa = _lns16_to_float_np(a_bits)
    fb = _lns16_to_float_np(b_bits)
    return _float_to_lns16_np(fa + fb)


def lns_dot_product(a_bits, b_bits):
    products = lns_logmul(
        _float_to_lns16_np(_lns16_to_float_np(a_bits)),
        _float_to_lns16_np(_lns16_to_float_np(b_bits))
    )
    result = _lns16_to_float_np(products).sum(axis=-1)
    return _float_to_lns16_np(result)


# ─── LNS16 class ─────────────────────────────────────────────────────────────

class LNS16:
    def __init__(self, data):
        if isinstance(data, torch.Tensor):
            self._data = data.float()
        elif isinstance(data, np.ndarray):
            self._data = torch.from_numpy(data.astype(np.float32))
        else:
            self._data = torch.tensor(data, dtype=torch.float32)

    def quantise(self):
        return LNS16(_quantise_torch(self._data))

    def __mul__(self, other):
        a = self._data.numpy()
        b = other._data.numpy()
        c = _lns16_to_float_np(lns_logmul(_float_to_lns16_np(a),
                                            _float_to_lns16_np(b)))
        return LNS16(torch.from_numpy(c.astype(np.float32)))

    def __add__(self, other):
        a = self._data.numpy()
        b = other._data.numpy()
        c = _lns16_to_float_np(lns_logadd(_float_to_lns16_np(a),
                                            _float_to_lns16_np(b)))
        return LNS16(torch.from_numpy(c.astype(np.float32)))

    def dot(self, other):
        a = self._data.numpy()
        b = other._data.numpy()
        c = _lns16_to_float_np(lns_dot_product(_float_to_lns16_np(a),
                                                 _float_to_lns16_np(b)))
        return LNS16(torch.from_numpy(c.astype(np.float32)))

    @property
    def tensor(self):
        return self._data

    def numpy(self):
        return self._data.cpu().numpy()

    def __repr__(self):
        return f"LNS16(shape={tuple(self._data.shape)})"

    @staticmethod
    def quantise_tensor(t: torch.Tensor) -> torch.Tensor:
        """Fast PyTorch-native LNS16 quantisation (no NumPy round-trip)."""
        return _quantise_torch(t)
