# Business Source License 1.1 - см. research/LICENSE
"""Живая шина: 7 процессов ОС, UDP на 127.0.0.1, реальное время 8 с на
синтетической траектории (в CI данных EuRoC нет). Пропускается без
librt_core.so. Проверяется только устойчивое: атакующий изолирован всеми
честными, у честных нет ложных тревог и ложных карантинов; в групповом
режиме узел с часами -200 мс выравнивается SYNC за <= 2 с и трафик узла
< 45 кбит/с."""
import os
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cryptography")

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
if RT not in sys.path:
    sys.path.insert(0, RT)

import rt_core                                               # noqa: E402

try:
    rt_core.load()
except OSError:
    pytest.skip("librt_core.so не собран или не грузится на этой платформе", allow_module_level=True)

import live_bus as lb                                        # noqa: E402

pytestmark = pytest.mark.concept


def test_live_bus_unicast_isolates_t8_without_false_alarms():
    cfg, res, pids = lb.run(None, duration_s=8.0, attack_s=3.0, loss=0.05, base_port=47400, mode="unicast")
    s = lb.summarize(cfg, res)
    assert len(set(pids)) == 7
    assert s["isolated"] == s["observers"] == 6
    assert max(s["iso_after_attack_s"]) < 2.0
    assert s["false_alarm"] == 0 and s["false_q"] == []
    assert s["honest_claims"] > 3000


def test_live_bus_multicast_aligns_lagging_clock():
    cfg, res, pids = lb.run(None, duration_s=8.0, attack_s=3.0, loss=0.05, base_port=47420,
                            group_port=47480, clock_off_ms={2: -200})
    s = lb.summarize(cfg, res)
    assert s["isolated"] == s["observers"] == 6               # и рассинхронизированный узел тоже
    assert s["false_alarm"] == 0 and s["false_q"] == []       # никто никого не отрезал зря
    assert s["desync_sync"][2]["aligned_s"] is not None and s["desync_sync"][2]["aligned_s"] <= 2.0
    assert s["residual_ms_max"] < 15.0
    assert s["kbit_peak"] < 45.0
