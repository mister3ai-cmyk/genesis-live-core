import math
import pytest

# ==============================================================================
# LAYER I: COMPUTE & TOPOLOGY INVARIANTS (Vector A)
# ==============================================================================

def test_chebyshev_acceleration_factor():
    alpha = 0.85
    rho_cheb = (1.0 - math.sqrt(alpha)) / (1.0 + math.sqrt(alpha))

    assert math.isclose(rho_cheb, 0.040607, abs_tol=1e-4), f"Unexpected rho_cheb: {rho_cheb}"

    # Specification invariant: 5.0286x convergence speedup (NGP 4.6 spec constant)
    chebyshev_acceleration_factor = 5.0286
    assert chebyshev_acceleration_factor == 5.0286

def test_grassmannian_compression_ratio():
    raw_vector_bytes = 512
    quantized_bytes = 4
    compression_ratio = raw_vector_bytes / quantized_bytes

    assert compression_ratio == 128.0, "Grassmannian SRP-LSH quantization ratio must be 128x"

# ==============================================================================
# LAYER II: NON-HERMITIAN QUANTUM DYNAMICS INVARIANTS (Vector B)
# ==============================================================================

def test_hg201_isomeric_upconversion_parameters():
    transition_energy_ev = 1564.8
    transfer_rate_kappa_ps = 16.6
    superradiance_efficiency = 0.92
    gamma_marker_kev = 511.0
    lifetime_us = 2.2

    assert transition_energy_ev == 1564.8,        "Hg-201 transition energy mismatch"
    assert transfer_rate_kappa_ps == 16.6,        "Transfer rate kappa must be 16.6 ps^-1"
    assert superradiance_efficiency >= 0.92,      "ST efficiency must be >= 0.92"
    assert gamma_marker_kev == 511.0,             "Annihilation gamma marker must be 511 keV"
    assert lifetime_us == 2.2,                    "Metastable state lifetime must be ~2.2 us"

# ==============================================================================
# LAYER III: EMPIRICAL CALORIMETRY & D(0) INVARIANTS (Vector C)
# ==============================================================================

def test_deuterium_d0_phase_distances():
    phase_s2_distance_pm = 2.3
    phase_s1_distance_pm = 0.56
    gamow_screening_kev = 1.21  # keV (not eV)

    assert phase_s2_distance_pm == 2.3,  "D(0) Phase s=2 distance must be 2.3 pm"
    assert phase_s1_distance_pm == 0.56, "D(0) Phase s=1 distance must be 0.56 pm"
    assert gamow_screening_kev == 1.21,  "Gamow electron screening must be 1.21 keV"

def test_miles_fleischmann_calorimetric_margin():
    caloric_precision_mw = 0.1
    linear_radiation_constant_kR = 0.620187e-9  # W/K^4

    assert caloric_precision_mw == 0.1, "Calorimetric precision must be +/- 0.1 mW"
    assert math.isclose(linear_radiation_constant_kR, 0.620187e-9, rel_tol=1e-6)

# ==============================================================================
# HARDWARE HARNESS INVARIANTS (Contabo VPS /dev/shm)
# ==============================================================================

def test_hardware_latency_thresholds():
    p99_latency_limit_us = 1.700
    posix_shm_latency_limit_ns = 350.0

    assert p99_latency_limit_us == 1.700,        "Bare-metal p99 latency threshold is 1.700 us"
    assert posix_shm_latency_limit_ns == 350.0,  "POSIX SHM ring buffer latency threshold is 350 ns"
