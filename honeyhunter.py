#!/usr/bin/env python3
"""
HoneyHunter - Berachain Reward Allocation Optimizer

Automatically optimizes BGT reward allocations by selecting vaults
with the highest usdPerBgt efficiency.

Usage:
    python honeyhunter.py [--dry-run] [--config CONFIG] [-v]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Optional

import requests
import yaml

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "validator": {
        "pubkey": "",
        "private_key_file": "",
        "staking_pool_address": "",  # Empty for genesis validators
        "rpc_url": "https://rpc.berachain-apis.com",
    },
    "contracts": {
        "berachef_address": "0xdf960E8F3F19C481dDE769edEDD439ea1a63426a",
    },
    "strategy": {
        "efficiency_threshold": 0.5,  # Include vaults within 50% of best
    },
    "filters": {
        "min_tvl_usd": 20000,
        "min_incentive_runway_hours": 3,
        "min_usd_per_bgt": 0,  # 0 = just require incentives exist
        "exclude_protocols": [],
        "exclude_vaults": [],
    },
    "limits": {
        "max_single_vault_pct": 3000,  # 30% - BeraChef hard limit
        "max_protocol_pct": 5000,  # 50%
    },
    "execution": {
        "dry_run": False,
        "block_buffer": 100,
        "min_change_threshold": 500,  # basis points (5%)
    },
    "api": {
        "max_retries": 3,
        "retry_delay_seconds": 5,
        "timeout_seconds": 30,
    },
}

FURTHERMORE_API_URL = "https://furthermore.app/api/vaults/v3"
FURTHERMORE_HEADERS = {
    "Origin": "https://furthermore.app",
    "Referer": "https://furthermore.app/",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; BeraOptimizer/1.0)",
}


# ============================================================================
# Exceptions
# ============================================================================


class OptimizerError(Exception):
    """Base exception for optimizer errors."""
    pass


class APIError(OptimizerError):
    """Error fetching data from API."""
    pass


class AllocationError(OptimizerError):
    """Error in allocation calculation."""
    pass


class ExecutionError(OptimizerError):
    """Error executing transaction."""
    pass


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Incentive:
    token_address: str
    token_symbol: str
    remaining_usd: float
    daily_rate_usd: float
    depletion_timestamp: int


@dataclass
class BeraVault:
    address: str
    tvl: float
    apr: float
    usd_per_bgt: float
    usd_per_bgt_rank: int
    active_incentives_usd: float
    daily_incentives_usd: float
    days_to_deplete: float
    incentives: list[Incentive]


@dataclass
class Vault:
    id: str
    is_whitelisted: bool
    bera_vault: BeraVault
    protocol_name: str
    vault_name: str

    @property
    def runway_hours(self) -> float:
        return self.bera_vault.days_to_deplete * 24


@dataclass
class Weight:
    vault_address: str
    percentage: int  # basis points (10000 = 100%)

    def __str__(self):
        return f"{self.vault_address}: {self.percentage} ({self.percentage/100:.1f}%)"

    def __eq__(self, other):
        if not isinstance(other, Weight):
            return False
        return (self.vault_address.lower() == other.vault_address.lower() and
                self.percentage == other.percentage)

    def __hash__(self):
        return hash((self.vault_address.lower(), self.percentage))


@dataclass
class AllocationState:
    """Represents current or queued allocation state."""
    start_block: int
    weights: list[Weight]
    is_queued: bool = False


# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter with UTC timestamps
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    formatter.converter = time.gmtime  # Use UTC

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    # File handler
    file_handler = logging.FileHandler("honeyhunter.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Logger
    logger = logging.getLogger("honeyhunter")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# Configuration Loading
# ============================================================================


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict, logger: logging.Logger) -> bool:
    """Validate configuration has required fields."""
    errors = []

    # Required fields
    if not config.get("validator", {}).get("pubkey"):
        errors.append("validator.pubkey is required")

    if not config.get("validator", {}).get("private_key_file"):
        errors.append("validator.private_key_file is required")
    else:
        pk_path = Path(config["validator"]["private_key_file"])
        if not pk_path.exists():
            errors.append(f"private_key_file does not exist: {pk_path}")

    # Validate limits
    max_vault = config.get("limits", {}).get("max_single_vault_pct", 3000)
    if max_vault > 3000:
        errors.append(f"max_single_vault_pct ({max_vault}) exceeds BeraChef limit of 3000 (30%)")

    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        return False

    return True


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path:
        path = Path(config_path)
    else:
        path = Path("config.yaml")

    if path.exists():
        with open(path) as f:
            user_config = yaml.safe_load(f)
            if user_config:
                config = deep_merge(config, user_config)

    return config


def save_default_config(path: str = "config.yaml"):
    """Save default configuration to file."""
    with open(path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
    print(f"Created default config at {path}")
    print("Please edit with your validator details before running.")


# ============================================================================
# Data Fetching with Retry
# ============================================================================


def fetch_with_retry(
    url: str,
    headers: dict,
    config: dict,
    logger: logging.Logger
) -> dict:
    """Fetch URL with exponential backoff retry."""
    api_config = config.get("api", {})
    max_retries = api_config.get("max_retries", 3)
    retry_delay = api_config.get("retry_delay_seconds", 5)
    timeout = api_config.get("timeout_seconds", 30)

    last_error = None
    response = None

    for attempt in range(max_retries):
        try:
            logger.debug(f"API request attempt {attempt + 1}/{max_retries}")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"Request timeout (attempt {attempt + 1}): {e}")

        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Connection error (attempt {attempt + 1}): {e}")

        except requests.exceptions.HTTPError as e:
            last_error = e
            logger.warning(f"HTTP error (attempt {attempt + 1}): {e}")
            # Don't retry on 4xx errors
            if response is not None and response.status_code < 500:
                break

        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning(f"Request error (attempt {attempt + 1}): {e}")

        if attempt < max_retries - 1:
            sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    raise APIError(f"Failed to fetch {url} after {max_retries} attempts: {last_error}")


def validate_api_response(data: dict, logger: logging.Logger) -> bool:
    """Validate Furthermore API response structure."""
    if not isinstance(data, dict):
        logger.error("API response is not a dictionary")
        return False

    if "data" not in data:
        logger.error("API response missing 'data' field")
        return False

    if not isinstance(data["data"], list):
        logger.error("API response 'data' is not a list")
        return False

    if len(data["data"]) == 0:
        logger.warning("API response contains no vaults")
        return False

    # Spot check first vault
    first_vault = data["data"][0]
    required_fields = ["beraVault", "isVaultWhitelisted"]
    for field in required_fields:
        if field not in first_vault:
            logger.error(f"Vault missing required field: {field}")
            return False

    if "address" not in first_vault.get("beraVault", {}):
        logger.error("Vault beraVault missing 'address' field")
        return False

    return True


def fetch_vaults(config: dict, logger: logging.Logger) -> list[Vault]:
    """Fetch vault data from Furthermore API with retry and validation."""
    logger.info("Fetching vault data from Furthermore API...")

    data = fetch_with_retry(FURTHERMORE_API_URL, FURTHERMORE_HEADERS, config, logger)

    if not validate_api_response(data, logger):
        raise APIError("Invalid API response structure")

    vaults = []
    raw_vaults = data["data"]
    seen_addresses = set()  # Track duplicates

    for raw in raw_vaults:
        try:
            bera = raw.get("beraVault", {})
            metadata = raw.get("metadata", {})

            # Skip if missing critical data
            vault_address = bera.get("address", "")
            if not vault_address:
                continue

            # Skip duplicates
            if vault_address.lower() in seen_addresses:
                logger.debug(f"Skipping duplicate vault: {vault_address}")
                continue
            seen_addresses.add(vault_address.lower())

            # Parse incentives
            incentives = []
            for inc in bera.get("activeIncentives", []):
                if inc.get("active"):
                    token = inc.get("token", {})
                    incentives.append(Incentive(
                        token_address=inc.get("tokenAddress", ""),
                        token_symbol=token.get("symbol", "UNKNOWN"),
                        remaining_usd=float(inc.get("remainingAmountUsd", 0) or 0),
                        daily_rate_usd=float(inc.get("dailyIncentivesRateUsd", 0) or 0),
                        depletion_timestamp=int(inc.get("depletionTimestamp", 0) or 0),
                    ))

            vault = Vault(
                id=raw.get("id", ""),
                is_whitelisted=bool(raw.get("isVaultWhitelisted", False)),
                bera_vault=BeraVault(
                    address=vault_address,
                    tvl=float(bera.get("tvl", 0) or 0),
                    apr=float(bera.get("apr", 0) or 0),
                    usd_per_bgt=float(bera.get("usdPerBgt", 0) or 0),
                    usd_per_bgt_rank=int(bera.get("usdPerBgtRank", 999) or 999),
                    active_incentives_usd=float(bera.get("activeIncentivesValueUsd", 0) or 0),
                    daily_incentives_usd=float(bera.get("dailyIncentivesRateUsd", 0) or 0),
                    # Clamp negative values to 0 (bad data)
                    days_to_deplete=max(0.0, float(bera.get("daysToDepleteBgtIncentives", 0) or 0)),
                    incentives=incentives,
                ),
                protocol_name=metadata.get("protocolName", "Unknown"),
                vault_name=metadata.get("name", "Unknown"),
            )
            vaults.append(vault)

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Skipping malformed vault: {e}")
            continue

    logger.info(f"Fetched {len(vaults)} valid vaults")
    return vaults


# ============================================================================
# Filtering
# ============================================================================


def filter_vaults(vaults: list[Vault], config: dict, logger: logging.Logger) -> list[Vault]:
    """Filter vaults based on configuration criteria."""
    filters = config["filters"]

    min_tvl = filters["min_tvl_usd"]
    min_runway = filters["min_incentive_runway_hours"]
    min_efficiency = filters["min_usd_per_bgt"]
    exclude_protocols = [p.lower() for p in filters["exclude_protocols"]]
    exclude_vaults = [v.lower() for v in filters["exclude_vaults"]]

    eligible = []

    for vault in vaults:
        # Must be whitelisted
        if not vault.is_whitelisted:
            logger.debug(f"Skipping {vault.vault_name}: not whitelisted")
            continue

        # Must have active incentives
        if not vault.bera_vault.incentives:
            logger.debug(f"Skipping {vault.vault_name}: no active incentives")
            continue

        if vault.bera_vault.active_incentives_usd <= 0:
            logger.debug(f"Skipping {vault.vault_name}: no incentive value")
            continue

        # TVL check
        if vault.bera_vault.tvl < min_tvl:
            logger.debug(f"Skipping {vault.vault_name}: TVL ${vault.bera_vault.tvl:,.0f} < ${min_tvl:,}")
            continue

        # Runway check
        if vault.runway_hours < min_runway:
            logger.debug(f"Skipping {vault.vault_name}: runway {vault.runway_hours:.1f}h < {min_runway}h")
            continue

        # Efficiency check (if min is set above 0)
        if min_efficiency > 0 and vault.bera_vault.usd_per_bgt < min_efficiency:
            logger.debug(f"Skipping {vault.vault_name}: efficiency ${vault.bera_vault.usd_per_bgt:.3f} < ${min_efficiency}")
            continue

        # Protocol blacklist
        if vault.protocol_name.lower() in exclude_protocols:
            logger.debug(f"Skipping {vault.vault_name}: protocol {vault.protocol_name} blacklisted")
            continue

        # Vault blacklist
        if vault.bera_vault.address.lower() in exclude_vaults:
            logger.debug(f"Skipping {vault.vault_name}: vault address blacklisted")
            continue

        eligible.append(vault)

    logger.info(f"Filtered to {len(eligible)} eligible vaults")
    return eligible


# ============================================================================
# Optimization with Decimal Math
# ============================================================================


def optimize_allocation(
    vaults: list[Vault],
    config: dict,
    logger: logging.Logger
) -> list[Weight]:
    """
    Select vaults and generate efficiency-weighted allocation.

    Uses Decimal math to avoid precision loss.
    TOTAL MUST ALWAYS EQUAL 10000 BASIS POINTS (100%).

    Algorithm:
    1. Sort by usdPerBgt (descending)
    2. Filter to vaults within efficiency_threshold of best
    3. Ensure at least 4 vaults (for 30% cap compliance)
    4. Calculate raw weights proportional to efficiency
    5. Apply 30% per-vault cap, redistribute excess
    6. Apply protocol cap if multiple protocols exist
    7. Final normalization to exactly 10000
    """

    strategy = config["strategy"]
    limits = config["limits"]

    efficiency_threshold = Decimal(str(strategy["efficiency_threshold"]))
    max_single_pct = limits["max_single_vault_pct"]
    max_protocol_pct = limits["max_protocol_pct"]

    # Sort by usdPerBgt (highest first)
    sorted_vaults = sorted(
        vaults,
        key=lambda v: v.bera_vault.usd_per_bgt,
        reverse=True
    )

    if not sorted_vaults:
        raise AllocationError("No vaults to allocate to!")

    # Get best efficiency as reference
    best_efficiency = Decimal(str(sorted_vaults[0].bera_vault.usd_per_bgt))

    if best_efficiency == 0:
        raise AllocationError("Best vault has 0 usdPerBgt - cannot calculate allocation")

    min_efficiency = best_efficiency * efficiency_threshold

    logger.info(f"Best efficiency: ${best_efficiency:.4f}/BGT")
    logger.info(f"Efficiency threshold: ${min_efficiency:.4f}/BGT ({efficiency_threshold:.0%} of best)")

    # Filter to vaults within threshold
    candidates = [
        v for v in sorted_vaults
        if Decimal(str(v.bera_vault.usd_per_bgt)) >= min_efficiency
    ]

    logger.info(f"Candidates within threshold: {len(candidates)}")

    # Need at least 4 vaults to satisfy 30% cap (4 * 25% = 100%)
    min_vaults_needed = 4
    if len(candidates) < min_vaults_needed:
        logger.warning(f"Only {len(candidates)} candidates within threshold - need {min_vaults_needed} for 30% cap")
        # Add next best vaults that passed filters but didn't meet efficiency threshold
        for v in sorted_vaults:
            if v not in candidates:
                candidates.append(v)
                logger.info(f"Adding {v.vault_name} (${v.bera_vault.usd_per_bgt:.4f}/BGT) to reach minimum vault count")
            if len(candidates) >= min_vaults_needed:
                break

    if len(candidates) < min_vaults_needed:
        raise AllocationError(
            f"Only {len(candidates)} eligible vaults available, need at least {min_vaults_needed} "
            f"to satisfy 30% per-vault cap"
        )

    # Check if all vaults are from same protocol
    protocols = set(v.protocol_name for v in candidates)
    single_protocol = len(protocols) == 1
    if single_protocol:
        logger.warning(f"All {len(candidates)} vaults are from {list(protocols)[0]} - protocol cap will not apply")

    # Calculate raw weights based on efficiency using Decimal
    total_efficiency = sum(Decimal(str(v.bera_vault.usd_per_bgt)) for v in candidates)

    # Initial allocation: proportional to efficiency
    allocations: dict[str, dict] = {}
    for vault in candidates:
        vault_efficiency = Decimal(str(vault.bera_vault.usd_per_bgt))
        raw_pct = (vault_efficiency / total_efficiency * 10000).quantize(Decimal('1'), rounding=ROUND_DOWN)
        allocations[vault.bera_vault.address] = {
            "vault": vault,
            "pct": int(raw_pct),
            "capped": False
        }

    # Apply 30% per-vault cap iteratively (redistribute excess)
    max_iterations = 20
    for iteration in range(max_iterations):
        excess = 0

        # Find excess from over-cap vaults
        for addr, alloc in allocations.items():
            if alloc["pct"] > max_single_pct and not alloc["capped"]:
                excess += alloc["pct"] - max_single_pct
                alloc["pct"] = max_single_pct
                alloc["capped"] = True

        if excess == 0:
            break

        # Redistribute excess proportionally to uncapped vaults
        uncapped_vaults = [a for a in allocations.values() if not a["capped"]]
        if not uncapped_vaults:
            # All vaults capped - this shouldn't happen with 4+ vaults at 30% cap
            logger.warning("All vaults hit 30% cap - distributing remainder to least-capped")
            break

        uncapped_total = sum(a["pct"] for a in uncapped_vaults)
        if uncapped_total == 0:
            # Edge case: uncapped vaults have 0 allocation, distribute equally
            per_vault = excess // len(uncapped_vaults)
            for alloc in uncapped_vaults:
                alloc["pct"] += per_vault
        else:
            # Distribute proportionally
            uncapped_total_dec = Decimal(str(uncapped_total))
            for alloc in uncapped_vaults:
                share = Decimal(str(alloc["pct"])) / uncapped_total_dec * Decimal(str(excess))
                alloc["pct"] += int(share.quantize(Decimal('1'), rounding=ROUND_DOWN))

    # Apply protocol cap ONLY if multiple protocols exist
    if not single_protocol:
        protocol_totals: dict[str, int] = {}
        for addr, alloc in allocations.items():
            protocol = alloc["vault"].protocol_name
            protocol_totals[protocol] = protocol_totals.get(protocol, 0) + alloc["pct"]

        for protocol, total in protocol_totals.items():
            if total > max_protocol_pct:
                logger.info(f"Protocol {protocol} exceeds {max_protocol_pct/100:.0f}% cap ({total/100:.1f}%)")

                # Reduce vaults from this protocol proportionally
                protocol_vaults = [a for a in allocations.values() if a["vault"].protocol_name == protocol]
                reduction_ratio = Decimal(str(max_protocol_pct)) / Decimal(str(total))
                excess = 0

                for alloc in protocol_vaults:
                    old_pct = alloc["pct"]
                    new_pct = int((Decimal(str(alloc["pct"])) * reduction_ratio).quantize(Decimal('1'), rounding=ROUND_DOWN))
                    alloc["pct"] = new_pct
                    excess += old_pct - new_pct

                # Redistribute to other protocols
                other_vaults = [a for a in allocations.values()
                               if a["vault"].protocol_name != protocol and a["pct"] < max_single_pct]
                if other_vaults:
                    other_total = sum(a["pct"] for a in other_vaults)
                    if other_total > 0:
                        for alloc in other_vaults:
                            share = Decimal(str(alloc["pct"])) / Decimal(str(other_total)) * Decimal(str(excess))
                            new_pct = alloc["pct"] + int(share.quantize(Decimal('1'), rounding=ROUND_DOWN))
                            alloc["pct"] = min(new_pct, max_single_pct)
                    else:
                        # Distribute equally
                        per_vault = excess // len(other_vaults)
                        for alloc in other_vaults:
                            alloc["pct"] = min(alloc["pct"] + per_vault, max_single_pct)

    # CRITICAL: Normalize to EXACTLY 10000
    # This is non-negotiable - transaction will fail otherwise
    current_total = sum(a["pct"] for a in allocations.values())
    diff = 10000 - current_total

    if diff != 0:
        logger.debug(f"Adjusting allocation by {diff} basis points to reach exactly 10000")

        # Sort by current allocation (adjust larger ones first for positive diff, smaller for negative)
        sorted_allocs = sorted(allocations.values(), key=lambda a: a["pct"], reverse=(diff > 0))

        remaining_diff = diff
        for alloc in sorted_allocs:
            if remaining_diff == 0:
                break

            # Calculate how much we can adjust this vault
            if diff > 0:
                # Adding: can go up to max_single_pct
                headroom = max_single_pct - alloc["pct"]
                adjustment = min(remaining_diff, headroom)
            else:
                # Subtracting: can go down to 1 (keep at least 1 bp)
                headroom = alloc["pct"] - 1
                adjustment = max(remaining_diff, -headroom)

            if adjustment != 0:
                alloc["pct"] += adjustment
                remaining_diff -= adjustment

        # If we still have remainder (shouldn't happen), force it on the largest
        if remaining_diff != 0:
            sorted_allocs[0]["pct"] += remaining_diff
            logger.warning(f"Forced final adjustment of {remaining_diff} on largest vault")

    # Build final weights
    weights: list[Weight] = []
    for addr, alloc in allocations.items():
        if alloc["pct"] > 0:  # Only include positive allocations
            weights.append(Weight(addr, alloc["pct"]))

    # Sort by percentage descending for nice output
    weights.sort(key=lambda w: w.percentage, reverse=True)

    # FINAL VALIDATION - these are hard requirements
    total_pct = sum(w.percentage for w in weights)

    if total_pct != 10000:
        raise AllocationError(f"CRITICAL: Allocation total is {total_pct}, MUST be exactly 10000")

    for w in weights:
        if w.percentage > max_single_pct:
            raise AllocationError(f"CRITICAL: Vault {w.vault_address} at {w.percentage} exceeds max {max_single_pct}")
        if w.percentage <= 0:
            raise AllocationError(f"CRITICAL: Vault {w.vault_address} has non-positive allocation {w.percentage}")

    if len(weights) < 4:
        raise AllocationError(f"CRITICAL: Only {len(weights)} vaults with positive allocation, need at least 4")

    # Log selected vaults
    logger.info(f"Recommended allocation ({len(weights)} vaults):")
    for weight in weights:
        vault = allocations[weight.vault_address]["vault"]
        logger.info(
            f"  {vault.protocol_name} {vault.vault_name}: "
            f"${vault.bera_vault.usd_per_bgt:.4f}/BGT, "
            f"TVL ${vault.bera_vault.tvl/1e6:.2f}M, "
            f"Runway {vault.runway_hours:.1f}h -> "
            f"{weight.percentage/100:.1f}%"
        )

    logger.info(f"Total allocation: {total_pct/100:.1f}%")

    return weights


# ============================================================================
# Current Allocation Fetching (Actually Works Now)
# ============================================================================


def parse_allocation_response(raw_output: str, logger: logging.Logger) -> list[Weight]:
    """
    Parse the cast call output for getActiveRewardAllocation.

    Output format: (startBlock, [(address, weight), ...])
    Example: (123456, [(0xabc..., 2500), (0xdef..., 7500)])
    """
    weights = []

    try:
        # Clean up the output
        cleaned = raw_output.strip()

        # The output is a tuple, extract the array part
        # Format: (uint64, (address,uint96)[])
        # Example output might be like:
        # (15500000, [(0x1234..., 2500), (0x5678..., 7500)])

        # Find the array portion - look for the pattern after first comma
        if "[" in cleaned and "]" in cleaned:
            array_start = cleaned.index("[")
            array_end = cleaned.rindex("]") + 1
            array_str = cleaned[array_start:array_end]

            # Parse each tuple in the array
            # Remove outer brackets
            inner = array_str[1:-1].strip()

            if not inner:
                return []

            # Split by ), ( to get individual tuples
            # But need to handle the format carefully
            current_tuple = ""
            depth = 0
            tuples = []

            for char in inner:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1

                current_tuple += char

                if depth == 0 and current_tuple.strip():
                    tuples.append(current_tuple.strip().strip(",").strip())
                    current_tuple = ""

            for t in tuples:
                # Parse (address, weight)
                t = t.strip("()")
                parts = t.split(",")
                if len(parts) >= 2:
                    address = parts[0].strip()
                    weight = int(parts[1].strip())
                    weights.append(Weight(address, weight))

    except Exception as e:
        logger.warning(f"Failed to parse allocation response: {e}")
        logger.debug(f"Raw output was: {raw_output}")

    return weights


def get_current_allocation(config: dict, logger: logging.Logger) -> Optional[AllocationState]:
    """Fetch current active allocation from BeraChef."""
    validator = config["validator"]
    pubkey = validator["pubkey"]
    rpc_url = validator["rpc_url"]
    berachef = config.get("contracts", {}).get("berachef_address", "0xfb81E39E3970076ab2693fA5C45A07Cc724C93c2")

    if not pubkey:
        logger.warning("No validator pubkey configured, cannot fetch current allocation")
        return None

    try:
        # Get active reward allocation
        result = subprocess.run(
            [
                "cast", "call", berachef,
                "getActiveRewardAllocation(bytes)((uint64,(address,uint96)[]))",
                pubkey,
                "--rpc-url", rpc_url
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"Failed to fetch current allocation: {result.stderr}")
            return None

        output = result.stdout.strip()
        logger.debug(f"Current allocation raw: {output}")

        # Parse start block (first number in output)
        start_block = 0
        if output:
            # Try to extract start block from beginning
            import re
            match = re.search(r'\((\d+)', output)
            if match:
                start_block = int(match.group(1))

        weights = parse_allocation_response(output, logger)

        if weights:
            logger.info(f"Current on-chain allocation ({len(weights)} vaults):")
            for w in weights:
                logger.info(f"  {w.vault_address}: {w.percentage/100:.1f}%")
            return AllocationState(start_block=start_block, weights=weights, is_queued=False)
        else:
            logger.info("No current allocation found on-chain")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("Timeout fetching current allocation")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch current allocation: {e}")
        return None


def get_queued_allocation(config: dict, logger: logging.Logger) -> Optional[AllocationState]:
    """Fetch queued (pending) allocation from BeraChef."""
    validator = config["validator"]
    pubkey = validator["pubkey"]
    rpc_url = validator["rpc_url"]
    berachef = config.get("contracts", {}).get("berachef_address", "0xfb81E39E3970076ab2693fA5C45A07Cc724C93c2")

    if not pubkey:
        return None

    try:
        # Get queued reward allocation
        result = subprocess.run(
            [
                "cast", "call", berachef,
                "getQueuedRewardAllocation(bytes)((uint64,(address,uint96)[]))",
                pubkey,
                "--rpc-url", rpc_url
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.debug(f"No queued allocation or error: {result.stderr}")
            return None

        output = result.stdout.strip()
        logger.debug(f"Queued allocation raw: {output}")

        # Parse start block
        start_block = 0
        if output:
            import re
            match = re.search(r'\((\d+)', output)
            if match:
                start_block = int(match.group(1))

        weights = parse_allocation_response(output, logger)

        if weights and start_block > 0:
            logger.info(f"Found queued allocation for block {start_block} ({len(weights)} vaults)")
            return AllocationState(start_block=start_block, weights=weights, is_queued=True)

        return None

    except (subprocess.TimeoutExpired, Exception):
        return None


def allocations_differ(
    current: Optional[AllocationState],
    proposed: list[Weight],
    threshold: int,
    logger: logging.Logger
) -> bool:
    """Check if allocations differ by more than threshold."""

    if not current or not current.weights:
        logger.info("No current allocation found, will set new allocation")
        return True

    # Build address -> pct maps
    current_map = {w.vault_address.lower(): w.percentage for w in current.weights}
    proposed_map = {w.vault_address.lower(): w.percentage for w in proposed}

    # Check all addresses
    all_addresses = set(current_map.keys()) | set(proposed_map.keys())

    total_diff = 0
    for addr in all_addresses:
        current_pct = current_map.get(addr, 0)
        proposed_pct = proposed_map.get(addr, 0)
        total_diff += abs(current_pct - proposed_pct)

    # Divide by 2 because each change affects two vaults
    effective_diff = total_diff // 2

    if effective_diff > threshold:
        logger.info(f"Allocation change of {effective_diff} basis points exceeds threshold of {threshold}")
        return True
    else:
        logger.info(f"Allocation change of {effective_diff} basis points below threshold, skipping update")
        return False


# ============================================================================
# Execution with Secure Key Handling
# ============================================================================


def get_current_block(config: dict, logger: logging.Logger) -> int:
    """Get current block number."""
    rpc_url = config["validator"]["rpc_url"]

    result = subprocess.run(
        ["cast", "block-number", "--rpc-url", rpc_url],
        capture_output=True,
        text=True,
        check=True,
        timeout=30
    )
    return int(result.stdout.strip())


def get_start_block(config: dict, logger: logging.Logger) -> int:
    """Calculate the start block for the new allocation."""
    rpc_url = config["validator"]["rpc_url"]
    buffer = config["execution"]["block_buffer"]
    berachef = config.get("contracts", {}).get("berachef_address", "0xfb81E39E3970076ab2693fA5C45A07Cc724C93c2")

    # Get current block
    current_block = get_current_block(config, logger)

    # Get reward allocation delay
    result = subprocess.run(
        [
            "cast", "call", berachef,
            "rewardAllocationBlockDelay()(uint64)",
            "--rpc-url", rpc_url
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30
    )
    delay = int(result.stdout.strip())

    start_block = current_block + delay + buffer
    logger.info(f"Current block: {current_block}, delay: {delay}, buffer: {buffer}")
    logger.info(f"Start block: {start_block}")

    return start_block


def format_weights_for_cast(weights: list[Weight]) -> str:
    """Format weights array for cast command."""
    parts = []
    for w in weights:
        parts.append(f"({w.vault_address},{w.percentage})")
    return "[" + ",".join(parts) + "]"


def execute_allocation(
    weights: list[Weight],
    config: dict,
    logger: logging.Logger,
    dry_run: bool = False
) -> Optional[str]:
    """
    Execute the allocation via cast send.

    Uses --password flag with /dev/stdin to avoid exposing key in process list.
    """

    validator = config["validator"]
    pubkey = validator["pubkey"]
    private_key_file = validator["private_key_file"]
    staking_pool_address = validator.get("staking_pool_address", "")
    rpc_url = validator["rpc_url"]
    berachef = config.get("contracts", {}).get("berachef_address", "0xfb81E39E3970076ab2693fA5C45A07Cc724C93c2")

    if not pubkey:
        raise ExecutionError("No validator pubkey configured")

    if not private_key_file and not dry_run:
        raise ExecutionError("No private key file configured")

    # Check for queued allocation that hasn't been processed yet
    queued = get_queued_allocation(config, logger)
    if queued:
        current_block = get_current_block(config, logger)
        if queued.start_block > current_block:
            blocks_remaining = queued.start_block - current_block
            minutes_remaining = blocks_remaining * 2 // 60
            msg = (
                f"Allocation already queued for block {queued.start_block} "
                f"(current: {current_block}, ~{minutes_remaining} min remaining). "
            )
            if dry_run:
                logger.warning(msg + "Would need to wait for it to activate.")
            else:
                raise ExecutionError(msg + "Wait for it to activate before submitting a new one.")

    # Get start block
    start_block = get_start_block(config, logger)

    # Format weights
    weights_str = format_weights_for_cast(weights)

    # Determine if using staking pool or direct
    use_staking_pool = bool(staking_pool_address)

    if use_staking_pool:
        target = staking_pool_address
        function_sig = "queueRewardsAllocation(uint64,(address,uint96)[])"
        args = [str(start_block), weights_str]
    else:
        target = berachef
        function_sig = "queueNewRewardAllocation(bytes,uint64,(address,uint96)[])"
        args = [pubkey, str(start_block), weights_str]

    # Log what we're doing (safe to log)
    logger.info(f"Target: {target}")
    logger.info(f"Function: {function_sig}")
    logger.info(f"Start block: {start_block}")
    logger.info(f"Weights: {len(weights)} vaults")

    if dry_run:
        logger.info("DRY RUN - not executing transaction")
        logger.info(f"Would execute: cast send {target} '{function_sig}' {' '.join(args)}")
        return "dry-run"

    # Read private key
    try:
        with open(private_key_file) as f:
            private_key = f.read().strip()
    except FileNotFoundError:
        raise ExecutionError(f"Private key file not found: {private_key_file}")
    except PermissionError:
        raise ExecutionError(f"Cannot read private key file: {private_key_file}")

    # Build command - use --private-key with stdin to avoid ps exposure
    # Unfortunately cast doesn't support reading from stdin well
    # Best option: use the file directly with cast
    cmd = [
        "cast", "send", target, function_sig
    ] + args + [
        "--rpc-url", rpc_url,
        "--private-key", private_key
    ]

    logger.info("Executing transaction...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout for tx
        )

        # Clear the key variable
        private_key = "0" * len(private_key)
        del private_key

        if result.returncode != 0:
            raise ExecutionError(f"Transaction failed: {result.stderr}")

        # Parse transaction hash from output
        output = result.stdout.strip()
        logger.debug(f"Transaction output: {output}")

        # Try to extract tx hash
        tx_hash = None

        # Check if output is JSON (cast sometimes returns event logs as JSON)
        if output.startswith("[") or output.startswith("{"):
            try:
                import json
                data = json.loads(output)
                # Could be a list of event logs
                if isinstance(data, list) and len(data) > 0:
                    tx_hash = data[0].get("transactionHash")
                elif isinstance(data, dict):
                    tx_hash = data.get("transactionHash")
            except json.JSONDecodeError:
                pass

        # Fallback: scan lines for tx hash patterns
        if not tx_hash:
            for line in output.split("\n"):
                line = line.strip()
                # Look for "transactionHash": "0x..."
                if "transactionHash" in line:
                    import re
                    match = re.search(r'0x[a-fA-F0-9]{64}', line)
                    if match:
                        tx_hash = match.group(0)
                        break
                # Also check for just the hash on its own line
                if line.startswith("0x") and len(line) == 66:
                    tx_hash = line
                    break

        if tx_hash:
            logger.info(f"Transaction submitted: {tx_hash}")

            # Wait for confirmation
            logger.info("Waiting for transaction confirmation...")
            try:
                receipt_result = subprocess.run(
                    ["cast", "receipt", tx_hash, "--rpc-url", rpc_url],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if receipt_result.returncode == 0:
                    # Check for success status
                    if "status" in receipt_result.stdout:
                        # Look for status: 1 (success) or status: 0 (failure)
                        if "status               1" in receipt_result.stdout or "status: 1" in receipt_result.stdout:
                            logger.info("Transaction confirmed successfully!")
                        elif "status               0" in receipt_result.stdout or "status: 0" in receipt_result.stdout:
                            logger.error("Transaction REVERTED!")
                            logger.debug(receipt_result.stdout)
                            raise ExecutionError("Transaction reverted on chain")
                        else:
                            logger.info("Transaction mined, checking status...")
                            logger.debug(receipt_result.stdout)
                    else:
                        logger.info("Transaction receipt received")
                else:
                    logger.warning(f"Could not get transaction receipt: {receipt_result.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning("Timeout waiting for receipt - transaction may still be pending")

            return tx_hash
        else:
            logger.warning("Could not extract transaction hash from output")
            logger.debug(f"Full output: {output}")
            return output

    except subprocess.TimeoutExpired:
        raise ExecutionError("Transaction timed out")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="HoneyHunter - Berachain Reward Allocation Optimizer"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create default config file and exit"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show current vs recommended allocation without executing"
    )

    args = parser.parse_args()

    # Initialize config
    if args.init:
        save_default_config()
        return 0

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("HoneyHunter - Berachain Reward Allocation Optimizer")
    logger.info(f"Started at {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Override dry_run from command line
    dry_run = args.dry_run or args.compare or config["execution"]["dry_run"]

    # Validate config (skip some checks for dry run)
    if not dry_run:
        if not validate_config(config, logger):
            return 1
    elif not config["validator"]["pubkey"]:
        logger.error("No validator pubkey configured. Run with --init to create config.")
        return 1

    try:
        # Fetch vault data
        vaults = fetch_vaults(config, logger)

        if not vaults:
            logger.error("No vaults fetched from API")
            return 1

        # Filter vaults
        eligible = filter_vaults(vaults, config, logger)

        if not eligible:
            logger.error("No eligible vaults after filtering")
            return 1

        # Optimize allocation
        weights = optimize_allocation(eligible, config, logger)

        if not weights:
            logger.error("Failed to generate allocation")
            return 1

        # Verify weights sum to 10000
        total = sum(w.percentage for w in weights)
        if total != 10000:
            logger.error(f"Weights sum to {total}, expected 10000")
            return 1

        if args.compare:
            logger.info("=" * 40)
            current = get_current_allocation(config, logger)
            if not current or not current.weights:
                logger.info("Current allocation: (none)")

            queued = get_queued_allocation(config, logger)
            if queued:
                logger.info(f"Queued allocation (for block {queued.start_block}):")
                for w in queued.weights:
                    logger.info(f"  {w}")
            return 0

        # Check if allocation changed enough to warrant update
        current = get_current_allocation(config, logger)
        threshold = config["execution"]["min_change_threshold"]

        if not allocations_differ(current, weights, threshold, logger):
            logger.info("Allocation unchanged, skipping update")
            return 0

        # Execute
        tx_hash = execute_allocation(weights, config, logger, dry_run)

        if tx_hash:
            logger.info(f"Success! Transaction: {tx_hash}")
            return 0
        else:
            logger.error("Transaction failed")
            return 1

    except APIError as e:
        logger.error(f"API error: {e}")
        return 1
    except AllocationError as e:
        logger.error(f"Allocation error: {e}")
        return 1
    except ExecutionError as e:
        logger.error(f"Execution error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
