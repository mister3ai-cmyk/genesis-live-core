// SPDX-License-Identifier: BSL-1.1
// Genesis Live Core — Genesis Tokenomics Pool
// Maksym Babych / Synapse Core Infrastructure © 2026
pragma solidity ^0.8.24;

/**
 * @title GenesisTokenomicsPool
 * @notice Канонический распределитель ликвидности Genesis Live.
 * @dev Маршрутизирует средства: 10% Автор, 20% DeSci Vault, 70% Tier 1 Stakers.
 *      Использует детерминированный паттерн rewardDebt для распределения дивидендов.
 */
contract GenesisTokenomicsPool {

    uint256 public constant AUTHOR_SHARE_BPS         = 1000;  // 10%
    uint256 public constant DESCIFOUNDATION_SHARE_BPS = 2000; // 20%
    uint256 public constant PLATFORM_SHARE_BPS        = 7000; // 70%
    uint256 public constant BPS_DENOMINATOR           = 10000;

    address public immutable owner;
    address public desciFoundationVault;
    address public gcbiEngine;

    // Tier 1 staking — Synthetix / MasterChef rewardDebt pattern
    mapping(address => uint256) public tier1StakedBalance;
    mapping(address => int256)  public rewardDebt;     // scaled 1e18
    mapping(address => uint256) public pendingRewards;

    uint256 public totalTier1Staked;
    uint256 public accumulatedPlatformPerShare; // scaled 1e18

    event FundsDistributed(
        uint256 totalAmount,
        address indexed author,
        uint256 authorAmount,
        uint256 foundationAmount,
        uint256 platformAmount
    );
    event Tier1Staked(address indexed holder, uint256 amount);
    event Tier1Unstaked(address indexed holder, uint256 amount);
    event RewardsClaimed(address indexed holder, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyEngine() {
        require(msg.sender == gcbiEngine, "Only GCBI Engine");
        _;
    }

    constructor(address _desciFoundationVault) {
        require(_desciFoundationVault != address(0), "Invalid vault address");
        owner = msg.sender;
        desciFoundationVault = _desciFoundationVault;
    }

    function setEngine(address _gcbiEngine) external onlyOwner {
        require(gcbiEngine == address(0), "Engine already set");
        require(_gcbiEngine != address(0), "Invalid engine address");
        gcbiEngine = _gcbiEngine;
    }

    /**
     * @notice Распределение средств при успешном завершении прогона.
     * @param author Адрес автора гипотезы (10% грант).
     */
    function distribute(address author) external payable onlyEngine {
        uint256 total = msg.value;
        require(total > 0,               "Zero value");
        require(author != address(0),    "Invalid author address");

        uint256 authorAmount     = (total * AUTHOR_SHARE_BPS)         / BPS_DENOMINATOR;
        uint256 foundationAmount = (total * DESCIFOUNDATION_SHARE_BPS) / BPS_DENOMINATOR;
        uint256 platformAmount   = total - authorAmount - foundationAmount;

        (bool ok1, ) = payable(author).call{value: authorAmount}("");
        require(ok1, "Author transfer failed");

        (bool ok2, ) = payable(desciFoundationVault).call{value: foundationAmount}("");
        require(ok2, "Foundation transfer failed");

        if (totalTier1Staked > 0 && platformAmount > 0) {
            accumulatedPlatformPerShare += (platformAmount * 1e18) / totalTier1Staked;
        } else if (platformAmount > 0) {
            // No stakers yet — platform reserve falls back to owner
            (bool ok3, ) = payable(owner).call{value: platformAmount}("");
            require(ok3, "Owner fallback transfer failed");
        }

        emit FundsDistributed(total, author, authorAmount, foundationAmount, platformAmount);
    }

    // ── Tier 1 Sovereign Foundry Pass staking ────────────────────────────────

    function stakeTier1() external payable {
        require(msg.value > 0, "Invalid stake amount");

        _updateRewards(msg.sender);

        tier1StakedBalance[msg.sender] += msg.value;
        totalTier1Staked              += msg.value;
        rewardDebt[msg.sender]        += int256((msg.value * accumulatedPlatformPerShare) / 1e18);

        emit Tier1Staked(msg.sender, msg.value);
    }

    function unstakeTier1(uint256 amount) external {
        require(tier1StakedBalance[msg.sender] >= amount, "Insufficient stake");

        _updateRewards(msg.sender);

        tier1StakedBalance[msg.sender] -= amount;
        totalTier1Staked              -= amount;
        rewardDebt[msg.sender]        -= int256((amount * accumulatedPlatformPerShare) / 1e18);

        (bool ok, ) = payable(msg.sender).call{value: amount}("");
        require(ok, "Unstake transfer failed");

        emit Tier1Unstaked(msg.sender, amount);
    }

    function claimRewards() external {
        _updateRewards(msg.sender);

        uint256 reward = pendingRewards[msg.sender];
        require(reward > 0, "No rewards available");

        pendingRewards[msg.sender] = 0;

        (bool ok, ) = payable(msg.sender).call{value: reward}("");
        require(ok, "Reward transfer failed");

        emit RewardsClaimed(msg.sender, reward);
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    function _updateRewards(address holder) internal {
        if (tier1StakedBalance[holder] > 0) {
            int256 accumulated = int256(
                (tier1StakedBalance[holder] * accumulatedPlatformPerShare) / 1e18
            );
            int256 pending = accumulated - rewardDebt[holder];
            if (pending > 0) {
                pendingRewards[holder] += uint256(pending);
                rewardDebt[holder]      = accumulated;
            }
        }
    }

    receive() external payable {}
}
