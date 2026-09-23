// SPDX-License-Identifier: BSL-1.1
// Genesis Live Core — GCBI Post-Factum Reward Engine
// Maksym Babych / Synapse Core Infrastructure © 2026
pragma solidity ^0.8.24;

import "./SiLA2HardwareVerifier.sol";

/**
 * @title GCBIPostFactumEngine
 * @notice Computes the Gross Civilization Benefit Index post-factum from
 *         hardware-verified telemetry and distributes stakes accordingly.
 *
 * GCBI = α·ΔHealthspan + β·ΔEnergy + γ·OpenAccess + δ·HardwareFeasibility
 *
 * All component scores are submitted by authorised oracle nodes and anchored
 * to a verified SiLA2 telemetry record — making them cryptographically bound
 * to physical instrument output. Plutocratic stake cannot fabricate a peak.
 *
 * Liquidity split on settlement:
 *   10% → hypothesis author
 *   20% → DeSci Foundation Vault
 *   70% → Platform operator
 *
 * Anti-Goodhart guarantee:
 *   GCBI is NOT scored before experiment execution.
 *   GCBI is NOT scored by NGP 4.5 predictions.
 *   GCBI is computed ONLY after telemetryId is verified on-chain.
 *   Tier-1 staking boosts are burned if physical peaks are absent.
 */
contract GCBIPostFactumEngine {

    // ── Constants ────────────────────────────────────────────────────────────

    // Basis points (10_000 = 100%)
    uint16 public constant AUTHOR_BPS     = 1000; // 10%
    uint16 public constant DESCI_BPS      = 2000; // 20%
    uint16 public constant PLATFORM_BPS   = 7000; // 70%

    // GCBI component weights (sum = 10_000 basis points)
    uint16 public constant ALPHA = 3500; // ΔHealthspan
    uint16 public constant BETA  = 3000; // ΔEnergy
    uint16 public constant GAMMA = 1500; // OpenAccess
    uint16 public constant DELTA = 2000; // HardwareFeasibility

    // Score range: 0–10_000 per component (fixed-point, 2 decimals)
    uint16 public constant SCORE_MAX = 10_000;

    // ── Data structures ──────────────────────────────────────────────────────

    struct HypothesisPool {
        address author;
        address payable deSciFundVault;
        address payable platformOperator;
        uint256 totalStake;          // wei accumulated in escrow
        bytes32 telemetryId;         // SiLA2 verified record anchor
        uint16  gcbiScore;           // 0–10_000 final composite score
        bool    settled;
    }

    struct ComponentScores {
        uint16 healthspan;
        uint16 energy;
        uint16 openAccess;
        uint16 feasibility;
    }

    // ── Storage ─────────────────────────────────────────────────────────────

    address public immutable admin;
    SiLA2HardwareVerifier public immutable verifier;

    mapping(uint256 => HypothesisPool)    public pools;
    mapping(uint256 => ComponentScores)   public scores;
    mapping(address => bool)              public authorisedOracles;

    uint256 public poolCount;

    // ── Events ───────────────────────────────────────────────────────────────

    event PoolCreated(uint256 indexed poolId, address indexed author, uint256 stake);
    event StakeAdded(uint256 indexed poolId, address indexed staker, uint256 amount);
    event GCBIScored(uint256 indexed poolId, bytes32 telemetryId, uint16 gcbiScore);
    event PoolSettled(uint256 indexed poolId, uint256 authorShare, uint256 desciShare, uint256 platformShare);
    event BoostBurned(uint256 indexed poolId, uint256 burnedAmount);

    // ── Constructor ──────────────────────────────────────────────────────────

    constructor(address verifierAddress) {
        admin    = msg.sender;
        verifier = SiLA2HardwareVerifier(verifierAddress);
    }

    // ── Admin ────────────────────────────────────────────────────────────────

    modifier onlyAdmin() {
        require(msg.sender == admin, "GCBI: not admin");
        _;
    }

    modifier onlyOracle() {
        require(authorisedOracles[msg.sender], "GCBI: not oracle");
        _;
    }

    function setOracle(address oracle, bool enabled) external onlyAdmin {
        authorisedOracles[oracle] = enabled;
    }

    // ── Pool lifecycle ────────────────────────────────────────────────────────

    /**
     * @notice Register a new hypothesis pool. Author receives 10% on settlement.
     */
    function createPool(
        address         author,
        address payable deSciFundVault,
        address payable platformOperator
    ) external onlyAdmin returns (uint256 poolId) {
        poolId = poolCount++;
        pools[poolId] = HypothesisPool({
            author:           author,
            deSciFundVault:   deSciFundVault,
            platformOperator: platformOperator,
            totalStake:       0,
            telemetryId:      bytes32(0),
            gcbiScore:        0,
            settled:          false
        });
        emit PoolCreated(poolId, author, 0);
    }

    /**
     * @notice Add stake (micro-bet) to an unsettled pool.
     */
    function addStake(uint256 poolId) external payable {
        HypothesisPool storage pool = pools[poolId];
        require(!pool.settled, "GCBI: pool settled");
        require(msg.value > 0, "GCBI: zero stake");
        pool.totalStake += msg.value;
        emit StakeAdded(poolId, msg.sender, msg.value);
    }

    // ── Post-factum scoring ───────────────────────────────────────────────────

    /**
     * @notice Oracle submits GCBI component scores anchored to a verified telemetry record.
     *         Scores are rejected if telemetryId is not on-chain verified.
     *
     * @param poolId       Target hypothesis pool.
     * @param telemetryId  SiLA2 record containing chromatographic + spectral peaks.
     * @param cs           Component scores (0–10_000 each).
     */
    function submitGCBI(
        uint256          poolId,
        bytes32          telemetryId,
        ComponentScores calldata cs
    ) external onlyOracle {
        HypothesisPool storage pool = pools[poolId];
        require(!pool.settled,                    "GCBI: already settled");
        require(pool.telemetryId == bytes32(0),   "GCBI: already scored");
        require(verifier.isVerified(telemetryId), "GCBI: telemetry not verified");

        _validateScores(cs);

        pool.telemetryId = telemetryId;
        scores[poolId]   = cs;

        uint16 composite = uint16(
            (uint32(cs.healthspan)  * ALPHA +
             uint32(cs.energy)      * BETA  +
             uint32(cs.openAccess)  * GAMMA +
             uint32(cs.feasibility) * DELTA) / 10_000
        );
        pool.gcbiScore = composite;

        emit GCBIScored(poolId, telemetryId, composite);
    }

    // ── Settlement ────────────────────────────────────────────────────────────

    /**
     * @notice Settle pool: distribute stakes according to canonical split.
     *         If gcbiScore == 0 (no peaks detected), burn Tier-1 boosts:
     *         stake is transferred to platform operator and DeSci vault only.
     */
    function settle(uint256 poolId) external onlyAdmin {
        HypothesisPool storage pool = pools[poolId];
        require(!pool.settled,               "GCBI: already settled");
        require(pool.telemetryId != bytes32(0), "GCBI: not scored yet");

        pool.settled = true;
        uint256 total = pool.totalStake;

        if (pool.gcbiScore == 0) {
            // Anti-Goodhart burn: no physical peaks → author grant zeroed
            uint256 desciShare    = total * DESCI_BPS    / 10_000;
            uint256 platformShare = total - desciShare;
            pool.deSciFundVault.transfer(desciShare);
            pool.platformOperator.transfer(platformShare);
            emit BoostBurned(poolId, total);
            emit PoolSettled(poolId, 0, desciShare, platformShare);
        } else {
            uint256 authorShare   = total * AUTHOR_BPS   / 10_000;
            uint256 desciShare    = total * DESCI_BPS    / 10_000;
            uint256 platformShare = total - authorShare - desciShare;
            payable(pool.author).transfer(authorShare);
            pool.deSciFundVault.transfer(desciShare);
            pool.platformOperator.transfer(platformShare);
            emit PoolSettled(poolId, authorShare, desciShare, platformShare);
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    function _validateScores(ComponentScores calldata cs) internal pure {
        require(cs.healthspan  <= SCORE_MAX, "GCBI: healthspan out of range");
        require(cs.energy      <= SCORE_MAX, "GCBI: energy out of range");
        require(cs.openAccess  <= SCORE_MAX, "GCBI: openAccess out of range");
        require(cs.feasibility <= SCORE_MAX, "GCBI: feasibility out of range");
    }

    // ── View helpers ─────────────────────────────────────────────────────────

    function getPool(uint256 poolId) external view returns (HypothesisPool memory) {
        return pools[poolId];
    }

    function getScores(uint256 poolId) external view returns (ComponentScores memory) {
        return scores[poolId];
    }
}
