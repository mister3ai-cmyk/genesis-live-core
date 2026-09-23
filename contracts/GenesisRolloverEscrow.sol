// SPDX-License-Identifier: BSL-1.1
// Genesis Live Core — Genesis Rollover Escrow
// Maksym Babych / Synapse Core Infrastructure © 2026
pragma solidity ^0.8.24;

import "./GCBIPostFactumEngine.sol";

/**
 * @title GenesisRolloverEscrow
 * @notice Manages the 10-slot hypothesis queue, collects micro-stakes,
 *         settles the winning hypothesis via GCBIPostFactumEngine, and
 *         rolls over the 9 unexecuted hypotheses into the next stream.
 *
 * Lifecycle per stream:
 *   1. Admin populates activeQueue[0..9] via addHypothesis().
 *   2. Participants micro-stake on any slot via stake(slotIndex).
 *   3. Admin marks one slot as winner via markExecuted(slotIndex, telemetryId).
 *      - Calls gcbiEngine.submitGCBI() through oracle, then gcbiEngine.settle().
 *   4. Admin calls rollOverUnexecuted() — the 9 remaining slots carry their
 *      pledgedPool and rolloverCount into the next stream's queue head.
 *
 * Carried Stake guarantee:
 *   - Balances of stakers on unexecuted slots are NEVER burned.
 *   - They roll over intact with the hypothesis slot (Zero Slippage).
 *   - rolloverCount increments per slot, allowing GCBI weight amplification.
 */
contract GenesisRolloverEscrow {

    // ── Constants ─────────────────────────────────────────────────────────────

    uint8 public constant QUEUE_SIZE = 10;

    // ── Data structures ───────────────────────────────────────────────────────

    struct HypothesisSlot {
        bytes32 taskSpecHash;   // keccak256 of the TaskSpec / InstrumentCard JSON
        address author;         // Researcher or institution address
        uint256 pledgedPool;    // Total micro-stake accumulated (wei)
        uint256 rolloverCount;  // Number of times carried to next stream
        uint256 gcbiPoolId;     // Corresponding pool ID in GCBIPostFactumEngine
        bool    executed;       // True once run in 3D Cube MODR cleanroom
        bool    active;         // True while occupying a queue slot
    }

    // Per-slot, per-staker balance — preserved across rollovers
    // slotIndex → staker → amount (wei)
    mapping(uint8 => mapping(address => uint256)) public stakerBalance;

    // ── Storage ───────────────────────────────────────────────────────────────

    address public immutable admin;
    GCBIPostFactumEngine public immutable gcbiEngine;

    // Fixed 10-slot active queue
    HypothesisSlot[QUEUE_SIZE] public activeQueue;

    // Overflow staging: hypotheses waiting to fill queue vacancies after rollover
    // Stored as a dynamic list; index 0 is next to enter the queue
    HypothesisSlot[] private _stagingQueue;

    uint256 public streamId;   // Increments each time rollOverUnexecuted() completes

    // ── Events ────────────────────────────────────────────────────────────────

    event HypothesisAdded(
        uint8   indexed slotIndex,
        bytes32 taskSpecHash,
        address author,
        uint256 gcbiPoolId
    );
    event Staked(uint8 indexed slotIndex, address indexed staker, uint256 amount);
    event SlotExecuted(uint8 indexed slotIndex, bytes32 telemetryId, uint16 gcbiScore);
    event RolledOver(
        uint8   indexed fromSlot,
        uint256 carriedStake,
        uint256 rolloverCount,
        uint256 newStreamId
    );
    event StreamAdvanced(uint256 indexed newStreamId);

    // ── Constructor ───────────────────────────────────────────────────────────

    constructor(address gcbiEngineAddress) {
        admin      = msg.sender;
        gcbiEngine = GCBIPostFactumEngine(payable(gcbiEngineAddress));
    }

    // ── Modifiers ─────────────────────────────────────────────────────────────

    modifier onlyAdmin() {
        require(msg.sender == admin, "Escrow: not admin");
        _;
    }

    // ── Queue population ──────────────────────────────────────────────────────

    /**
     * @notice Add a hypothesis to a specific queue slot.
     *         Slot must be vacant (not active).
     * @param slotIndex    0–9 position in activeQueue.
     * @param taskSpecHash keccak256 of the off-chain TaskSpec JSON.
     * @param author       Researcher wallet.
     * @param gcbiPoolId   Pre-created pool ID in GCBIPostFactumEngine.
     */
    function addHypothesis(
        uint8   slotIndex,
        bytes32 taskSpecHash,
        address author,
        uint256 gcbiPoolId
    ) external onlyAdmin {
        require(slotIndex < QUEUE_SIZE,        "Escrow: invalid slot");
        require(!activeQueue[slotIndex].active, "Escrow: slot occupied");

        activeQueue[slotIndex] = HypothesisSlot({
            taskSpecHash:  taskSpecHash,
            author:        author,
            pledgedPool:   0,
            rolloverCount: 0,
            gcbiPoolId:    gcbiPoolId,
            executed:      false,
            active:        true
        });

        emit HypothesisAdded(slotIndex, taskSpecHash, author, gcbiPoolId);
    }

    // ── Staking ───────────────────────────────────────────────────────────────

    /**
     * @notice Micro-stake on a hypothesis slot ($1–$5 equivalent in ETH/L2).
     *         Funds are held in escrow; forwarded to GCBIPostFactumEngine pool.
     */
    function stake(uint8 slotIndex) external payable {
        require(slotIndex < QUEUE_SIZE,            "Escrow: invalid slot");
        require(activeQueue[slotIndex].active,      "Escrow: slot not active");
        require(!activeQueue[slotIndex].executed,   "Escrow: slot already executed");
        require(msg.value > 0,                      "Escrow: zero stake");

        activeQueue[slotIndex].pledgedPool += msg.value;
        stakerBalance[slotIndex][msg.sender] += msg.value;

        // Forward stake into the GCBI pool so the engine holds the funds
        gcbiEngine.addStake{value: msg.value}(activeQueue[slotIndex].gcbiPoolId);

        emit Staked(slotIndex, msg.sender, msg.value);
    }

    // ── Execution & settlement ────────────────────────────────────────────────

    /**
     * @notice Mark a slot as executed (ran in 3D Cube MODR cleanroom).
     *         Submits GCBI scores and triggers settlement in GCBIPostFactumEngine.
     *
     * @param slotIndex    The winning hypothesis slot.
     * @param telemetryId  Verified SiLA2 telemetry record from hardware run.
     * @param cs           GCBI component scores from authorised oracle.
     */
    function markExecuted(
        uint8                                 slotIndex,
        bytes32                               telemetryId,
        GCBIPostFactumEngine.ComponentScores calldata cs
    ) external onlyAdmin {
        require(slotIndex < QUEUE_SIZE,           "Escrow: invalid slot");
        require(activeQueue[slotIndex].active,     "Escrow: slot not active");
        require(!activeQueue[slotIndex].executed,  "Escrow: already executed");

        activeQueue[slotIndex].executed = true;

        uint256 poolId = activeQueue[slotIndex].gcbiPoolId;

        // Submit GCBI scores (this contract must be an authorised oracle)
        gcbiEngine.submitGCBI(poolId, telemetryId, cs);

        // Retrieve composite score for event
        GCBIPostFactumEngine.HypothesisPool memory pool = gcbiEngine.getPool(poolId);

        // Settle: distribute 10/20/70 (or burn if gcbiScore == 0)
        gcbiEngine.settle(poolId);

        emit SlotExecuted(slotIndex, telemetryId, pool.gcbiScore);
    }

    // ── Rollover ──────────────────────────────────────────────────────────────

    /**
     * @notice Roll over the 9 unexecuted hypotheses into the next stream.
     *         - Each unexecuted slot's rolloverCount is incremented.
     *         - pledgedPool is preserved (Zero Slippage for stakers).
     *         - The executed slot is cleared; vacated positions fill from _stagingQueue.
     *         - streamId is incremented.
     *
     * Must be called after exactly one slot has been executed this stream.
     */
    function rollOverUnexecuted() external onlyAdmin {
        uint8 executedCount = 0;
        for (uint8 i = 0; i < QUEUE_SIZE; i++) {
            if (activeQueue[i].active && activeQueue[i].executed) {
                executedCount++;
            }
        }
        require(executedCount == 1, "Escrow: need exactly 1 executed slot");

        uint256 nextStream = streamId + 1;

        for (uint8 i = 0; i < QUEUE_SIZE; i++) {
            HypothesisSlot storage slot = activeQueue[i];
            if (!slot.active) continue;

            if (slot.executed) {
                // Clear the executed slot; fill from staging if available
                _clearSlot(i);
                if (_stagingQueue.length > 0) {
                    activeQueue[i] = _stagingQueue[0];
                    _removeFirstStaging();
                }
            } else {
                // Carry forward: increment rollover counter, preserve pledgedPool
                slot.rolloverCount += 1;
                emit RolledOver(i, slot.pledgedPool, slot.rolloverCount, nextStream);
            }
        }

        streamId = nextStream;
        emit StreamAdvanced(nextStream);
    }

    /**
     * @notice Stage a hypothesis to enter the queue on the next rollover.
     *         Used when all 10 slots are occupied but new proposals arrive.
     */
    function stageHypothesis(
        bytes32 taskSpecHash,
        address author,
        uint256 gcbiPoolId
    ) external onlyAdmin {
        _stagingQueue.push(HypothesisSlot({
            taskSpecHash:  taskSpecHash,
            author:        author,
            pledgedPool:   0,
            rolloverCount: 0,
            gcbiPoolId:    gcbiPoolId,
            executed:      false,
            active:        true
        }));
    }

    // ── View helpers ──────────────────────────────────────────────────────────

    function getSlot(uint8 slotIndex) external view returns (HypothesisSlot memory) {
        require(slotIndex < QUEUE_SIZE, "Escrow: invalid slot");
        return activeQueue[slotIndex];
    }

    function stagingQueueLength() external view returns (uint256) {
        return _stagingQueue.length;
    }

    function getStakerBalance(uint8 slotIndex, address staker) external view returns (uint256) {
        return stakerBalance[slotIndex][staker];
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    function _clearSlot(uint8 i) internal {
        delete activeQueue[i];
    }

    function _removeFirstStaging() internal {
        uint256 len = _stagingQueue.length;
        for (uint256 j = 0; j < len - 1; j++) {
            _stagingQueue[j] = _stagingQueue[j + 1];
        }
        _stagingQueue.pop();
    }
}
