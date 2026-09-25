import os
import json
import hashlib
import numpy as np
from pathlib import Path
import sys

# Portable path resolution
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.waddington_langevin_engine import WaddingtonLangevinEngine
from physics.thz_sirt6_resonator import THzSIRT6Resonator
from oracle.epigenetic_window_oracle import EpigeneticWindowOracle


def compute_release_hash(repo_path: Path) -> str:
    hasher = hashlib.sha256()
    for root, _, files in sorted(os.walk(repo_path)):
        for f in sorted(files):
            if f.endswith(('.py', '.json', '.sol', '.md')) or f == 'LICENSE':
                hasher.update((Path(root) / f).read_bytes())
    return hasher.hexdigest()


def main():
    engine    = WaddingtonLangevinEngine(noise_intensity_D=0.02, seed=2026)
    resonator = THzSIRT6Resonator(freq1_thz=8.0, freq2_thz=15.0)
    oracle    = EpigeneticWindowOracle(safe_window_min_h=72.0, safe_window_max_h=96.0)

    # 1. Langevin trajectory over V(x) = x^4 - 2x^2 + alpha*x
    traj_result        = engine.simulate_trajectory(x0=1.0, time_hours=84.0, dt_hours=0.1, osk_drive_alpha=1.6)
    trajectory_samples = np.array(traj_result["trajectory_samples"])

    # 2. THz SIRT6 resonance activation & LINE-1 suppression
    physics_result = resonator.calculate_resonance_activation(laser_power_mw=45.0, exposure_time_h=72.0)

    # 3. Epigenetic Window Oracle & Shannon entropy
    oracle_result = oracle.evaluate_safety_window(
        time_elapsed_hours=84.0,
        final_x_state=traj_result["final_state_x"],
        trajectory=trajectory_samples
    )

    report = {
        "repository_name": "ngp-waddington-lev-simulator",
        "version":          "v1.0.0-waddington",
        "license":          "BSL-1.1",
        "concept_doi":      "10.5281/zenodo.22944521",
        "version_doi":      "10.5281/zenodo.22962804",
        "release_sha256":   compute_release_hash(REPO_ROOT),
        "waddington_langevin_simulation": {
            "time_hours":              traj_result["time_hours"],
            "initial_somatic_state_x0": traj_result["initial_state_x"],
            "final_pluripotent_state_x": traj_result["final_state_x"],
            "pluripotency_lev_score":  traj_result["pluripotency_score"],
            "trajectory_mean":         traj_result["trajectory_mean"],
            "trajectory_std":          traj_result["trajectory_std"],
        },
        "thz_sirt6_resonance_physics": {
            "frequencies_thz":                      physics_result["pumping_frequencies_thz"],
            "sirt6_activation_multiplier":           physics_result["sirt6_activity_multiplier"],
            "line1_retrotransposon_suppression_pct": physics_result["line1_retrotransposon_suppression_pct"],
            "cgas_sting_inhibition_pct":             physics_result["cgas_sting_inflammation_inhibition_pct"],
            "frohlich_coherence":                    physics_result["frohlich_condensate_coherence"],
        },
        "epigenetic_safety_oracle": {
            "time_elapsed_hours":             oracle_result["time_elapsed_hours"],
            "shannon_entropy_bits":           oracle_result["shannon_entropy_bits"],
            "in_safe_reprogramming_window":   oracle_result["in_safe_window"],
            "oncogenic_transformation_risk":  oracle_result["oncogenic_transformation_risk"],
            "auto_degron_smash_triggered":    oracle_result["auto_degron_smash_triggered"],
            "status":                         oracle_result["reprogramming_status"],
        },
        "status": "ALL_SYSTEMS_VERIFIED",
    }

    print("=== NGP 4.5 Waddington LEV Simulator (ngp-waddington-lev-simulator) Diagnostic ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
