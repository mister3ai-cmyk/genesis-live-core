// SPDX-License-Identifier: BSL-1.1
// Genesis Live Core — SiLA 2 Hardware Telemetry Verifier
// Maksym Babych / Synapse Core Infrastructure © 2026
pragma solidity ^0.8.24;

/**
 * @title SiLA2HardwareVerifier
 * @notice On-chain verification of cryptographically signed instrument telemetry
 *         from the 3D Cube MODR cleanroom (Waters ACQUITY UPLC, Hamilton STARlet,
 *         Hamamatsu spectrometer). Device keys are ECDSA secp256k1.
 *
 * Trust model:
 *   - Each instrument has a registered deviceAddress (Ethereum-compatible pubkey).
 *   - A run produces a TelemetryRecord: Merkle root of raw sensor frames + metadata.
 *   - The record is submitted with an ECDSA signature from the instrument's key.
 *   - Only verified records are eligible for GCBI post-factum scoring.
 */
contract SiLA2HardwareVerifier {

    // ── Data structures ─────────────────────────────────────────────────────

    struct TelemetryRecord {
        bytes32 merkleRoot;      // Merkle root of raw sensor frame hashes
        address deviceAddress;   // Instrument's registered signing key
        uint32  runId;           // Monotonic run counter per device
        uint64  timestamp;       // Unix epoch at record submission
        bool    verified;
    }

    // ── Storage ─────────────────────────────────────────────────────────────

    address public immutable admin;

    // deviceAddress → registered (whitelisted instruments)
    mapping(address => bool) public registeredDevices;

    // telemetryId → record
    mapping(bytes32 => TelemetryRecord) public records;

    // runId nonce per device to prevent replay
    mapping(address => uint32) public lastRunId;

    // ── Events ───────────────────────────────────────────────────────────────

    event DeviceRegistered(address indexed device);
    event DeviceRevoked(address indexed device);
    event TelemetryVerified(
        bytes32 indexed telemetryId,
        address indexed device,
        uint32  runId,
        bytes32 merkleRoot
    );

    // ── Constructor ──────────────────────────────────────────────────────────

    constructor() {
        admin = msg.sender;
    }

    // ── Admin ────────────────────────────────────────────────────────────────

    modifier onlyAdmin() {
        require(msg.sender == admin, "SiLA2: not admin");
        _;
    }

    function registerDevice(address device) external onlyAdmin {
        registeredDevices[device] = true;
        emit DeviceRegistered(device);
    }

    function revokeDevice(address device) external onlyAdmin {
        registeredDevices[device] = false;
        emit DeviceRevoked(device);
    }

    // ── Core verification ────────────────────────────────────────────────────

    /**
     * @notice Submit and verify a signed telemetry record from an instrument.
     * @param merkleRoot  Merkle root of the instrument's raw sensor frame batch.
     * @param device      Instrument address (signing key).
     * @param runId       Monotonic run counter — must be strictly greater than lastRunId.
     * @param timestamp   Unix timestamp of measurement.
     * @param v,r,s       ECDSA signature components from instrument key over telemetryId.
     * @return telemetryId Unique identifier for this verified record.
     */
    function submitTelemetry(
        bytes32 merkleRoot,
        address device,
        uint32  runId,
        uint64  timestamp,
        uint8   v,
        bytes32 r,
        bytes32 s
    ) external returns (bytes32 telemetryId) {
        require(registeredDevices[device], "SiLA2: device not registered");
        require(runId > lastRunId[device], "SiLA2: replay detected");

        telemetryId = keccak256(abi.encodePacked(merkleRoot, device, runId, timestamp));

        // Recover signer from Ethereum-prefixed hash
        bytes32 ethHash = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", telemetryId)
        );
        address signer = ecrecover(ethHash, v, r, s);
        require(signer == device, "SiLA2: invalid instrument signature");

        lastRunId[device] = runId;

        records[telemetryId] = TelemetryRecord({
            merkleRoot:    merkleRoot,
            deviceAddress: device,
            runId:         runId,
            timestamp:     timestamp,
            verified:      true
        });

        emit TelemetryVerified(telemetryId, device, runId, merkleRoot);
    }

    /**
     * @notice Check whether a telemetry record exists and is verified.
     */
    function isVerified(bytes32 telemetryId) external view returns (bool) {
        return records[telemetryId].verified;
    }

    /**
     * @notice Verify that a specific sensor frame hash is included in a record's
     *         Merkle tree (caller provides proof).
     * @param telemetryId  The verified record to check against.
     * @param frameHash    Leaf hash of the individual sensor frame.
     * @param proof        Merkle proof path (sibling hashes, left-to-right order).
     * @param leafIndex    0-based index of the leaf in the tree.
     */
    function verifyFrame(
        bytes32   telemetryId,
        bytes32   frameHash,
        bytes32[] calldata proof,
        uint256   leafIndex
    ) external view returns (bool) {
        require(records[telemetryId].verified, "SiLA2: record not found");
        bytes32 root = records[telemetryId].merkleRoot;
        return _verifyMerkle(frameHash, proof, leafIndex, root);
    }

    // ── Internal Merkle verification ─────────────────────────────────────────

    function _verifyMerkle(
        bytes32   leaf,
        bytes32[] calldata proof,
        uint256   index,
        bytes32   expectedRoot
    ) internal pure returns (bool) {
        bytes32 computed = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            if (index % 2 == 0) {
                computed = keccak256(abi.encodePacked(computed, proof[i]));
            } else {
                computed = keccak256(abi.encodePacked(proof[i], computed));
            }
            index /= 2;
        }
        return computed == expectedRoot;
    }
}
