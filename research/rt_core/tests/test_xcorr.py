# Business Source License 1.1 - см. research/LICENSE
"""rt_xcorr: сверка с numpy, синтетика без шума позы, EuRoC (если данные
есть: research/rt_core/data/euroc/). Пропускается без librt_core.so."""
import os
import sys

import pytest

np = pytest.importorskip("numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
if RT not in sys.path:
    sys.path.insert(0, RT)

import rt_core                                               # noqa: E402

try:
    LIB = rt_core.load()
except OSError:
    pytest.skip("librt_core.so не собран или не грузится на этой платформе", allow_module_level=True)

import xcorr_check as xc                                     # noqa: E402

pytestmark = pytest.mark.concept
DATA = os.path.join(RT, "data", "euroc")


def _ref_lag(a, b, L):
    """Эталон: целый сдвиг с максимумом многоканального Пирсона."""
    n = len(a)
    best, kb = -2, 0
    for k in range(-L, L + 1):
        lo, hi = max(0, -k), min(n, n - k)
        x, y = a[lo:hi], b[lo + k:hi + k]
        x, y = x - x.mean(0), y - y.mean(0)
        r = np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))
        if r > best:
            best, kb = r, k
    return kb, best


def test_integer_peak_matches_numpy_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = np.cumsum(rng.normal(size=(400, 3)), 0)
        s = int(rng.integers(-15, 16))
        b = np.roll(a, s, axis=0) + rng.normal(0, 0.5, (400, 3))
        rc, lag, peak = rt_core.xcorr(LIB, a, b, 20, 1e-6)
        kb, best = _ref_lag(a, b, 20)
        assert rc == rt_core.OK
        assert round(lag) == kb and abs(peak - best) < 1e-9


def test_low_excitation_on_hover():
    a = np.full((400, 3), 0.01) + np.random.default_rng(1).normal(0, 1e-3, (400, 3))
    assert rt_core.xcorr(LIB, a, a, 20, xc.MIN_VAR)[0] == rt_core.ERR_LOW_EXCITATION


@pytest.mark.skipif(not os.path.isdir(DATA), reason="данных EuRoC нет")
def test_euroc_injected_delay_within_one_sample():
    for seq in sorted(os.listdir(DATA)):
        name, res = xc.euroc(LIB, os.path.join(DATA, seq))
        assert abs(res["natural"]) < 1.0                       # IMU и истина выровнены < 1 мс
        for dms in xc.DELAYS_MS:
            r = res[dms]
            assert r["within"] >= 95.0, (name, dms, r)         # <= 5 мс (1 отсчёт)
            assert r["low"] <= 0.05 * r["n"]
