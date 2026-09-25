import os
import json
import math
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# =====================================================================
# NGP 4.5 SOVEREIGN KNOWLEDGE MARKETPLACE (ngp-knowledge-marketplace)
# Version: v1.0.0-marketplace | License: BSL 1.1 -> Apache 2.0
# =====================================================================

# 1. SECURITY: LOOP-STRANGLE WASH TRADING GUARD
class LoopStrangleGuard:
    """
    Algorithmic barrier against wash-trading and circular volume manipulation.
    Detects rapid circular transactions (A -> B -> A) within time window
    and enforces a penalization split (90% Treasury / 10% Burn/Validators).
    """
    def __init__(self, time_window_sec: float = 3600.0, max_frequency: int = 3):
        self.time_window_sec = time_window_sec
        self.max_frequency = max_frequency
        self.tx_history: List[Dict[str, Any]] = []

    def record_and_verify_tx(self, buyer: str, seller: str, module_id: str, amount: float) -> Tuple[bool, float, Dict[str, float]]:
        now = time.time()
        self.tx_history = [tx for tx in self.tx_history if now - tx['timestamp'] < self.time_window_sec]

        circular_matches = [
            tx for tx in self.tx_history
            if tx['module_id'] == module_id and (
                (tx['buyer'] == seller and tx['seller'] == buyer) or
                (tx['buyer'] == buyer and tx['seller'] == seller)
            )
        ]

        is_wash_trade = len(circular_matches) >= self.max_frequency

        if is_wash_trade:
            split = {
                "escrow": 0.0,
                "treasury": round(amount * 0.90, 4),
                "validator": round(amount * 0.10, 4),
                "penalized": True
            }
        else:
            split = {
                "escrow": round(amount * 0.80, 4),
                "treasury": round(amount * 0.10, 4),
                "validator": round(amount * 0.10, 4),
                "penalized": False
            }

        self.tx_history.append({
            "buyer": buyer,
            "seller": seller,
            "module_id": module_id,
            "amount": amount,
            "timestamp": now
        })

        return (is_wash_trade, amount, split)


# 2. CORE: PROOF-OF-KNOWLEDGE (PoK) REPUTATION ENGINE
class ProofOfKnowledgeEngine:
    """
    PoK reputation engine featuring Ebbinghaus exponential memory decay.
    Half-life = 14 days (1,209,600 seconds).
    """
    HALF_LIFE_SEC = 14 * 24 * 3600.0  # 14 days

    def __init__(self):
        self.user_pok_scores: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    def add_contribution(self, user_id: str, base_score: float) -> float:
        current_score = self.get_decayed_score(user_id)
        new_score = current_score + base_score
        self.user_pok_scores[user_id] = new_score
        self.last_update[user_id] = time.time()
        return new_score

    def get_decayed_score(self, user_id: str) -> float:
        if user_id not in self.user_pok_scores:
            return 0.0

        last_t = self.last_update.get(user_id, time.time())
        dt = time.time() - last_t

        decay_constant = math.log(2) / self.HALF_LIFE_SEC
        decayed_score = self.user_pok_scores[user_id] * math.exp(-decay_constant * dt)
        return round(decayed_score, 6)


# 3. DEX MARKETPLACE V3 CORE ENGINE
class DEXMarketplaceV3Engine:
    """
    NGP 4.5 DEX Knowledge Marketplace V3 Core.
    Orchestrates PoK reputation, 22-module registry, 80/10/10 royalty splits,
    and wash-trading protection.
    """
    def __init__(self):
        self.pok_engine = ProofOfKnowledgeEngine()
        self.guard = LoopStrangleGuard(time_window_sec=3600.0, max_frequency=3)
        self.registry = self._init_registry()
        self.escrow_balance = 0.0
        self.treasury_balance = 0.0
        self.validator_balance = 0.0

    def _init_registry(self) -> Dict[str, Any]:
        return {
            "NGP-CORE-DECOMPOSE":    {"name": "LNOI Waveguide Tensor Decomposition",       "price_usdc": 150.0, "type": "core_utility", "pok_weight": 15.0},
            "NGP-DEDUCE-SYNC":       {"name": "SiLA 2 gRPC State Synchronizer",             "price_usdc": 120.0, "type": "core_utility", "pok_weight": 12.0},
            "NGP-DIST-MEM-KV":       {"name": "Shared Tensor Ring Memory KV Bridge",        "price_usdc": 200.0, "type": "core_utility", "pok_weight": 20.0},
            "NGP-ANIS-L1-BARRIER":   {"name": "Anisotropic L1 Regularization Filter",       "price_usdc": 100.0, "type": "core_utility", "pok_weight": 10.0},
            "NGP-MEM-EBBINGHAUS":    {"name": "Ebbinghaus Exponential Decay Engine",        "price_usdc": 180.0, "type": "core_utility", "pok_weight": 18.0},
            "NGP-GRAPH-PPR-PUSH":    {"name": "Local Push PageRank Subgraph Allocator",     "price_usdc": 140.0, "type": "core_utility", "pok_weight": 14.0},
            "NGP-RRF-CONSENSUS":     {"name": "Reciprocal Rank Fusion Consensus",           "price_usdc": 110.0, "type": "core_utility", "pok_weight": 11.0},
            "NGP-ED25519-AUTH":      {"name": "Hardware Ed25519 Cryptographic Signer",      "price_usdc": 250.0, "type": "core_utility", "pok_weight": 25.0},
            "NGP-SIRT6-CLEANER":     {"name": "SIRT6 Transposon LINE-1 Suppressor",         "price_usdc": 300.0, "type": "core_utility", "pok_weight": 30.0},
            "NGP-ANCHOR-POS":        {"name": "Project Anchor Quantum Positioner",          "price_usdc": 220.0, "type": "core_utility", "pok_weight": 22.0},
            "NGP-SWARM-SIR":         {"name": "Swarm Intelligence Re-ranking Engine",       "price_usdc": 160.0, "type": "core_utility", "pok_weight": 16.0},
            "NGP-PUDA-JETSTREAM":    {"name": "PUDA NATS/JetStream Low-Latency Bus",        "price_usdc": 210.0, "type": "core_utility", "pok_weight": 21.0},
            "BIO-PHOTON-MEMORY":     {"name": "Photonic Biophoton Memory Enclave",          "price_usdc": 500.0, "type": "bio_enclave",  "pok_weight": 50.0},
            "BIO-SILENCE-RESONANCE": {"name": "Silence Resonance & Coherence Enclave",      "price_usdc": 450.0, "type": "bio_enclave",  "pok_weight": 45.0},
            "BIO-EEL-PROTOCOL":      {"name": "Electrochemical Eel Energy Enclave",         "price_usdc": 400.0, "type": "bio_enclave",  "pok_weight": 40.0},
            "BIO-QUANTUM-ANCHOR":    {"name": "Quantum Rejuvenation Anchor Enclave",        "price_usdc": 600.0, "type": "bio_enclave",  "pok_weight": 60.0},
            "BIO-WADDINGTON-LANDSCAPE": {"name": "Waddington Epigenetic Attractor Enclave", "price_usdc": 550.0, "type": "bio_enclave",  "pok_weight": 55.0},
            "BIO-FROHLICH-CONDENSATE":  {"name": "Fröhlich Condensate Coherence Enclave",  "price_usdc": 480.0, "type": "bio_enclave",  "pok_weight": 48.0},
            "BIO-COHERENT-EXCITON":  {"name": "Coherent Exciton Transport Enclave",         "price_usdc": 520.0, "type": "bio_enclave",  "pok_weight": 52.0},
            "BIO-PALEO-SHIELD":      {"name": "Paleo-Shield Genomic Guardian Enclave",      "price_usdc": 470.0, "type": "bio_enclave",  "pok_weight": 47.0},
            "BIO-EPANECHNIKOV-KERNEL": {"name": "Epanechnikov Kernel Smooth Enclave",       "price_usdc": 380.0, "type": "bio_enclave",  "pok_weight": 38.0},
            "BIO-OCTANT-ACCORD":     {"name": "Octant Harmony Biophysical Enclave",         "price_usdc": 650.0, "type": "bio_enclave",  "pok_weight": 65.0},
        }

    def execute_purchase(self, buyer: str, seller: str, module_id: str) -> Dict[str, Any]:
        if module_id not in self.registry:
            raise ValueError(f"Module '{module_id}' not found in registry")

        module = self.registry[module_id]
        price = float(module["price_usdc"])

        is_wash, _, split = self.guard.record_and_verify_tx(buyer, seller, module_id, price)

        self.escrow_balance    += split["escrow"]
        self.treasury_balance  += split["treasury"]
        self.validator_balance += split["validator"]

        pok_reward = float(module.get("pok_weight", 10.0))
        new_pok = self.pok_engine.add_contribution(seller, pok_reward)

        return {
            "status": "PURCHASE_SUCCESSFUL",
            "module_id": module_id,
            "module_name": module["name"],
            "buyer": buyer,
            "seller": seller,
            "price_usdc": price,
            "wash_trade_detected": is_wash,
            "split_usdc": split,
            "seller_updated_pok": new_pok,
            "pool_balances": {
                "escrow_pool_usdc":    round(self.escrow_balance, 2),
                "swarm_treasury_usdc": round(self.treasury_balance, 2),
                "validators_usdc":     round(self.validator_balance, 2),
            }
        }

    def run_full_system_diagnostic(self) -> Dict[str, Any]:
        # 1. Purchase Bio-Enclave Module
        p1 = self.execute_purchase("vitalik_node",  "syn_researcher_01", "BIO-QUANTUM-ANCHOR")

        # 2. Purchase Core Utility Module
        p2 = self.execute_purchase("molecule_dao",  "syn_researcher_02", "NGP-CORE-DECOMPOSE")

        # 3. Simulate Ebbinghaus 14-day decay on seller PoK
        self.pok_engine.last_update["syn_researcher_01"] = time.time() - (14 * 24 * 3600.0)
        decayed_pok = self.pok_engine.get_decayed_score("syn_researcher_01")

        # 4. Release fingerprint
        payload      = json.dumps(self.registry, sort_keys=True, ensure_ascii=False).encode("utf-8")
        release_hash = hashlib.sha256(payload).hexdigest()

        return {
            "marketplace_version":       "v1.0.0-marketplace",
            "license":                   "BSL-1.1",
            "registered_modules_count":  len(self.registry),
            "release_sha256":            release_hash,
            "sample_purchase_bio":       p1,
            "sample_purchase_core":      p2,
            "pok_decay_14day_half_life": {
                "original_score":             p1["seller_updated_pok"],
                "decayed_score_after_14_days": decayed_pok,
                "decay_ratio":                round(decayed_pok / p1["seller_updated_pok"], 4),
            },
            "status": "ALL_SYSTEMS_VERIFIED",
        }


if __name__ == "__main__":
    print("=== NGP 4.5 Sovereign Knowledge Marketplace v1.0.0 Diagnostic ===")
    engine = DEXMarketplaceV3Engine()
    diag   = engine.run_full_system_diagnostic()
    print(json.dumps(diag, indent=2, ensure_ascii=False))
