# Business Source License 1.1 - см. research/LICENSE
"""EuRoC MAV. Без данных: прогон конвейера euroc_bench на синтетических CSV
в формате EuRoC (только механика, не результат). С данными в
research/rt_core/data/euroc/<SEQ>/ (euroc_fetch.py): целостность потоков."""
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

import euroc_bench as eb                                     # noqa: E402
import sixdof as sd                                          # noqa: E402

pytestmark = pytest.mark.concept
DATA = os.path.join(RT, "data", "euroc")


def _write_synthetic(dirpath, seed):
    """Синтетика в формате EuRoC: state_groundtruth_estimate0 (200 Гц, 17
    колонок) и imu0 (200 Гц: w, a), из генератора 6-DoF."""
    rng = np.random.default_rng(seed)
    tr = sd.truth(rng, "turn", T=40.0)
    i = np.arange(0, len(tr["t"]), 5)
    t_ns = (tr["t"][i] * 1e9).astype(np.int64) + 1_403_715_000_000_000_000
    vel = np.gradient(tr["pos"], tr["dt"], axis=0)[i]
    gt = np.zeros((len(i), 17))
    gt[:, 1:4] = tr["pos"][i]
    gt[:, 4] = 1.0
    gt[:, 8:11] = vel
    tk, f, w = sd.imu(rng, tr)
    ti_ns = (tk * 1e9).astype(np.int64) + 1_403_715_000_000_000_000
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "state_groundtruth_estimate0.csv"), "w") as fh:
        fh.write("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bw_x,bw_y,bw_z,ba_x,ba_y,ba_z\n")
        for k in range(len(i)):
            fh.write("%d," % t_ns[k] + ",".join("%.6f" % x for x in gt[k, 1:]) + "\n")
    with open(os.path.join(dirpath, "imu0.csv"), "w") as fh:
        fh.write("#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n")
        for k in range(len(tk)):
            fh.write("%d," % ti_ns[k] + ",".join("%.6f" % x for x in np.concatenate([w[k], f[k]])) + "\n")


def test_pipeline_runs_on_synthetic_euroc_format(tmp_path, capsys):
    a, b = tmp_path / "SYN_A", tmp_path / "SYN_B"
    _write_synthetic(str(a), 1)
    _write_synthetic(str(b), 2)
    seqs = [eb.load(str(a)), eb.load(str(b))]
    r = eb.gate_run(LIB, seqs[0], 1.0, 15.0, 1.0, 28 * eb.SIGMA, "T8")
    assert r["hon"] > 100 and r["att"] > 50
    eb.task_b(LIB, seqs)
    out = capsys.readouterr().out
    assert "1. Сдвиг во времени" in out and "3. Нарушение связности" in out


@pytest.mark.skipif(not os.path.isdir(DATA), reason="данных EuRoC нет: python3 euroc_fetch.py --seq ...")
def test_real_euroc_streams_are_consistent():
    seqs = [os.path.join(DATA, s) for s in sorted(os.listdir(DATA))]
    assert seqs
    for p in seqs:
        d = eb.load(p)
        assert d["t"][-1] > 30.0                                  # сессии EuRoC - минуты
        assert 150 < 1 / np.median(np.diff(d["t"])) < 250         # истина ~200 Гц
        assert 150 < 1 / np.median(np.diff(d["ti"])) < 250        # IMU 200 Гц
        g = np.median(np.linalg.norm(d["a"], axis=1))
        assert 9.0 < g < 10.6                                     # удельная сила ~ g
