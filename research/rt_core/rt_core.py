# Business Source License 1.1 - см. research/LICENSE
"""ctypes-обёртка librt_core.so (Linux/WSL). Путь к библиотеке - переменная
RT_CORE_LIB или librt_core.so рядом с этим файлом. Память контекста выделяет
Python (буфер rt_ctx_size() байт), C не вызывает malloc."""
import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PASS, REFUTE, FAILSAFE = 0, 1, 2
OK, NO_PRED = 0, 1


class Cfg(ctypes.Structure):
    _fields_ = [("horizon_us", ctypes.c_int64), ("max_skew_us", ctypes.c_int64),
                ("drop_tol", ctypes.c_double), ("thr", ctypes.c_double),
                ("pos_tol", ctypes.c_double), ("min_pts", ctypes.c_int)]


def load(path=None):
    """Загрузить библиотеку; OSError, если её нет или платформа не та."""
    lib = ctypes.CDLL(path or os.environ.get("RT_CORE_LIB") or os.path.join(HERE, "librt_core.so"))
    D, I, I64 = ctypes.c_double, ctypes.c_int, ctypes.c_int64
    P, DP = ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
    lib.rt_ctx_size.restype = ctypes.c_size_t
    lib.rt_cfg_default.argtypes = [ctypes.POINTER(Cfg)]
    lib.rt_init.argtypes = [P, ctypes.POINTER(Cfg)]
    lib.rt_kin_push.argtypes = [P, I, I, I64, I64, DP]
    lib.rt_kin_predict.argtypes = [P, I, I64, I, DP]
    lib.rt_kin_gate.argtypes = [P, I, I, I64, I64, DP, DP, DP]
    lib.rt_gr_orth_mgs.argtypes = [I, I, DP, DP]
    lib.rt_gr_chordal2.argtypes = [I, I, DP, DP]
    lib.rt_gr_chordal2.restype = D
    lib.rt_xcorr_delay.argtypes = [DP, DP, I, I, D, DP, DP]
    lib.rt_xcorr_delay_multi.argtypes = [DP, DP, I, I, I, D, DP, DP]
    lib.rt_kin_forget_author.argtypes = [P, I, I]
    return lib


ERR_LOW_EXCITATION = -4


def xcorr(lib, a, b, max_lag, min_var):
    """(код, задержка b относительно a в отсчётах, пик корреляции).
    a, b - (n,) или (n, каналы)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    n, ch = a.shape
    ac = np.ascontiguousarray(a.T).ravel()            # каналы подряд
    bc = np.ascontiguousarray(b.T).ravel()
    lag, peak = ctypes.c_double(0.0), ctypes.c_double(0.0)
    p = ctypes.POINTER(ctypes.c_double)
    rc = lib.rt_xcorr_delay_multi(ac.ctypes.data_as(p), bc.ctypes.data_as(p), n, ch, max_lag, min_var,
                                  ctypes.byref(lag), ctypes.byref(peak))
    return rc, lag.value, peak.value


def _vec(x, n=3):
    a = (ctypes.c_double * n)()
    for i in range(n):
        a[i] = float(x[i]) if i < len(x) else 0.0
    return a


class Kin:
    """Контекст шлюза: push / predict / gate. Времена - микросекунды."""

    def __init__(self, lib, **cfg):
        self.lib = lib
        self.buf = ctypes.create_string_buffer(lib.rt_ctx_size())
        c = Cfg()
        lib.rt_cfg_default(ctypes.byref(c))
        for k, v in cfg.items():
            setattr(c, k, v)
        assert lib.rt_init(self.buf, ctypes.byref(c)) == OK

    def push(self, target, author, t_us, p, now_us=None):
        return self.lib.rt_kin_push(self.buf, target, author, int(t_us),
                                    int(t_us if now_us is None else now_us), _vec(p))

    def predict(self, target, t_us, exclude=-1):
        out = _vec((0, 0, 0))
        rc = self.lib.rt_kin_predict(self.buf, target, int(t_us), exclude, out)
        return (out[0], out[1], out[2]) if rc == OK else None

    def gate(self, target, author, t_claim_us, now_us, claim, mine):
        r = ctypes.c_double(0.0)
        v = self.lib.rt_kin_gate(self.buf, target, author, int(t_claim_us), int(now_us),
                                 _vec(claim), _vec(mine), ctypes.byref(r))
        return v, r.value


def _col(A):
    A = np.asarray(A, dtype=np.float64)
    return np.ascontiguousarray(A.T).ravel(), A.shape          # по столбцам


def orth(lib, A):
    a, (n, k) = _col(A)
    q = np.empty_like(a)
    rc = lib.rt_gr_orth_mgs(n, k, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc != OK:
        raise ValueError("rt_gr_orth_mgs: %d" % rc)
    return q.reshape(k, n).T


def chordal2(lib, U, V):
    u, (n, k) = _col(U)
    v, _ = _col(V)
    p = ctypes.POINTER(ctypes.c_double)
    return lib.rt_gr_chordal2(n, k, u.ctypes.data_as(p), v.ctypes.data_as(p))
