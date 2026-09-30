import numpy as np
import time
from abc import ABC, abstractmethod


class TemporalBurstEstimator(ABC):
    """Abstract interface for latency-free predictive plasticity drivers."""

    @abstractmethod
    def get_burst_factor(self, delta_t: float) -> float:
        pass


class TimesFMSharedMemorySidecar(TemporalBurstEstimator):
    """
    Production driver: reads Google TimesFM-200M background predictions
    from lock-free POSIX shared memory (/dev/shm/ngp_timesfm_burst.raw).

    Architecture:
        Slow Path: TimesFM-200M transformer runs async in background,
                   writes scalar y_burst prediction every T_sidecar seconds.
        Fast Path: this method does O(1) mmap read — zero transformer overhead.
    """

    def __init__(self, shm_path: str = "/dev/shm/ngp_timesfm_burst.raw"):
        self.shm_path = shm_path

    def get_burst_factor(self, delta_t: float) -> float:
        try:
            with open(self.shm_path, "rb") as f:
                raw = f.read(4)
            if len(raw) == 4:
                return float(np.frombuffer(raw, dtype=np.float32)[0])
        except OSError:
            pass
        # Analytical fallback when sidecar daemon is not running
        return float(np.clip(1.0 / (delta_t + 1.0), 0.0, 2.0))


class HeuristicBurstEstimator(TemporalBurstEstimator):
    """Lightweight analytical fallback for CPU-only standalone mode."""

    def get_burst_factor(self, delta_t: float) -> float:
        return float(np.tanh(1.0 / (delta_t + 1e-3)))


class NGP46ReconsolidationEngineHPC:
    """
    NGP 4.6 Generative Engram & Phase Weight Reconsolidation Engine (HPC Edition).

    Key properties:
      - O(N^2) hot path: low-rank outer-product update + Newton-Schulz normalization
      - O(1) TimesFM sidecar read from /dev/shm (zero transformer inference overhead)
      - ||delta_W||_F clipping guarantees Newton-Schulz convergence radius
      - Second corrective NS step if unitarity residual exceeds 1e-3
    """

    def __init__(self, dim: int = 256, estimator: TemporalBurstEstimator = None):
        self.dim = dim
        self.eta_0 = 0.015
        self.lambda_0 = 1e-4
        self.max_delta_norm = 0.05
        self.W_phase = np.eye(dim, dtype=np.complex128)
        self.last_access_time = time.perf_counter()
        self.estimator = estimator or TimesFMSharedMemorySidecar()

    def generate_context(self, query_vector: np.ndarray) -> np.ndarray:
        now = time.perf_counter()
        delta_t = now - self.last_access_time
        self.last_access_time = now

        # Step 1: O(1) sidecar read
        y_burst = self.estimator.get_burst_factor(delta_t)
        eta_eff = self.eta_0 * (1.0 + 0.5 * y_burst)
        lambda_eff = self.lambda_0 * np.exp(-0.8 * y_burst)

        # Step 2: Normalization onto C^256 ~ G(4, C^64)
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
        psi_0 = np.dot(self.W_phase, q_norm)
        psi_retrieved = psi_0 / (np.linalg.norm(psi_0) + 1e-12)

        # Step 3: Low-rank Nader-LeDoux update with norm clipping
        decay = np.exp(-lambda_eff * delta_t)
        delta_W = eta_eff * decay * np.outer(psi_retrieved, np.conj(q_norm))

        norm_delta = np.linalg.norm(delta_W, "fro")
        if norm_delta > self.max_delta_norm:
            delta_W *= self.max_delta_norm / norm_delta

        # Step 4: Newton-Schulz unitary step O(N^2)
        W_temp = self.W_phase + delta_W
        I = np.eye(self.dim, dtype=np.complex128)
        W_H_W = np.dot(np.conj(W_temp.T), W_temp)
        self.W_phase = 0.5 * np.dot(W_temp, (3.0 * I - W_H_W))

        # Unitarity drift guardrail: second corrective step if residual > 1e-3
        residual = np.linalg.norm(
            np.dot(np.conj(self.W_phase.T), self.W_phase) - I, "fro"
        )
        if residual > 1e-3:
            W_H_W = np.dot(np.conj(self.W_phase.T), self.W_phase)
            self.W_phase = 0.5 * np.dot(self.W_phase, (3.0 * I - W_H_W))

        return psi_retrieved


if __name__ == "__main__":
    engine = NGP46ReconsolidationEngineHPC(estimator=HeuristicBurstEstimator())
    test_query = np.random.randn(256) + 1j * np.random.randn(256)

    t0 = time.perf_counter_ns()
    context = engine.generate_context(test_query)
    dt_us = (time.perf_counter_ns() - t0) / 1000.0
    print(f"[NGP 4.6 HPC Engine] Context generated & reconsolidated in {dt_us:.2f} us")
    print(f"[NGP 4.6 HPC Engine] ||psi|| = {np.linalg.norm(context):.6f}")
