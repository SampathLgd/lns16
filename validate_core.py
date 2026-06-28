"""
Numpy-only validation of LNS16 core math.
Run this to verify encode/decode, logmul, logadd, dot product.
"""
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
INT_BITS   = 6
FRAC_BITS  = 9
FRAC_SCALE = 2 ** FRAC_BITS        # 512
INT_BIAS   = 2 ** (INT_BITS - 1)   # 32
ZERO_SENTINEL = np.uint16(0xFFFF)


def float_to_lns16(x):
    x = np.asarray(x, dtype=np.float64)
    zero_mask = (x == 0.0)
    sign_mask = (x < 0)
    abs_x = np.abs(x)
    safe  = np.where(zero_mask, 1.0, abs_x)
    log2  = np.log2(safe)
    fixed = np.round(log2 * FRAC_SCALE).astype(np.int32)
    ip    = np.clip(fixed >> FRAC_BITS, -INT_BIAS, INT_BIAS - 1)
    fp    = fixed & (FRAC_SCALE - 1)
    ib    = (ip + INT_BIAS).astype(np.uint16)
    s     = sign_mask.astype(np.uint16)
    packed = (s << 15) | (ib << FRAC_BITS) | fp.astype(np.uint16)
    return np.where(zero_mask, ZERO_SENTINEL, packed).astype(np.uint16)


def lns16_to_float(bits):
    bits = np.asarray(bits, dtype=np.uint16)
    zero = (bits == ZERO_SENTINEL)
    s    = ((bits >> 15) & 1).astype(np.float64)
    ib   = ((bits >> FRAC_BITS) & ((1 << INT_BITS) - 1)).astype(np.int32)
    fr   = (bits & (FRAC_SCALE - 1)).astype(np.float64)
    ip   = ib - INT_BIAS
    log2 = ip.astype(np.float64) + fr / FRAC_SCALE
    mag  = np.exp2(log2)
    out  = np.where(s == 0, mag, -mag)
    return np.where(zero, 0.0, out)


def logmul(a, b):
    def lf(x):
        ib = ((x >> FRAC_BITS) & ((1 << INT_BITS) - 1)).astype(np.int32)
        fr = (x & (FRAC_SCALE - 1)).astype(np.int32)
        return ((ib - INT_BIAS) << FRAC_BITS) | fr
    za = (a == ZERO_SENTINEL); zb = (b == ZERO_SENTINEL)
    log_c = lf(a) + lf(b)
    s  = ((a >> 15) ^ (b >> 15)).astype(np.uint16)
    ic = np.clip(log_c >> FRAC_BITS, -INT_BIAS, INT_BIAS - 1)
    fc = (log_c & (FRAC_SCALE - 1)).astype(np.uint16)
    packed = (s << 15) | ((ic + INT_BIAS).astype(np.uint16) << FRAC_BITS) | fc
    return np.where(za | zb, ZERO_SENTINEL, packed).astype(np.uint16)


def logadd(a, b):
    return float_to_lns16(lns16_to_float(a) + lns16_to_float(b))


def dot(a, b):
    return float_to_lns16(np.sum(lns16_to_float(logmul(float_to_lns16(lns16_to_float(a)),
                                                         float_to_lns16(lns16_to_float(b))))))


# ─── Tests ────────────────────────────────────────────────────────────────────

PASS = "✓"; FAIL = "✗"

def check(name, got, expected, rtol=0.01):
    if expected == 0:
        ok = abs(got) < 0.05
    else:
        ok = abs(got - expected) / abs(expected) < rtol
    status = PASS if ok else FAIL
    print(f"  [{status}] {name:<40} got={got:10.5f}  expected={expected:10.5f}")
    return ok


def run_validation():
    print("=" * 65)
    print("  LNS16 Core Math Validation")
    print("=" * 65)
    passed = 0; total = 0

    print("\n  --- Encode/Decode ---")
    for v in [1.0, -1.0, 0.5, -0.5, 4.0, 0.125, 0.0, 100.0, -3.14]:
        bits  = float_to_lns16(np.array([v]))
        recon = lns16_to_float(bits)[0]
        total += 1
        passed += check(f"round-trip({v})", recon, v)

    print("\n  --- logmul ---")
    mul_cases = [(2, 3, 6), (-2, 3, -6), (-2, -3, 6), (4, 0.5, 2), (0, 5, 0)]
    for a, b, e in mul_cases:
        ab = float_to_lns16(np.array([a])); bb = float_to_lns16(np.array([b]))
        res = lns16_to_float(logmul(ab, bb))[0]
        total += 1; passed += check(f"logmul({a},{b})", res, e)

    print("\n  --- logadd ---")
    add_cases = [(2, 3, 5), (-1, -2, -3), (0, 4, 4), (3, -3, 0), (10, 0.5, 10.5)]
    for a, b, e in add_cases:
        ab = float_to_lns16(np.array([a])); bb = float_to_lns16(np.array([b]))
        res = lns16_to_float(logadd(ab, bb))[0]
        total += 1; passed += check(f"logadd({a},{b})", res, e, rtol=0.02)

    print("\n  --- dot product ---")
    dp_cases = [([1,2,3],[4,5,6],32), ([1,0],[0,1],0), ([2,2],[3,3],12)]
    for a, b, e in dp_cases:
        ab = float_to_lns16(np.array(a, dtype=np.float64))
        bb = float_to_lns16(np.array(b, dtype=np.float64))
        prods = logmul(ab, bb)
        res = lns16_to_float(float_to_lns16(np.array([lns16_to_float(prods).sum()])))[0]
        total += 1; passed += check(f"dot({a},{b})", res, e, rtol=0.02)

    print("\n" + "=" * 65)
    print(f"  Results: {passed}/{total} passed")
    print("=" * 65)
    return passed == total


if __name__ == "__main__":
    ok = run_validation()
    exit(0 if ok else 1)
