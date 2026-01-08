# 🍯 HoneyHunter

Automated BGT reward allocation optimizer for Berachain validators. Hunts for the highest-yielding vaults based on `usdPerBgt` efficiency.

## How It Works

1. **Fetches** vault data from the Furthermore API (with retry logic)
2. **Filters** vaults by: whitelisted, has active incentives, TVL, runway, blacklists
3. **Sorts** by `usdPerBgt` - USD value of incentives per BGT directed
4. **Weights** allocation proportionally to efficiency (higher = more allocation)
5. **Caps** at 30% per vault (BeraChef rule), 50% per protocol
6. **Normalizes** to exactly 100% (required for transaction success)
7. **Executes** via `cast send` with transaction confirmation

## Prerequisites

- Python 3.11+
- [Foundry](https://book.getfoundry.sh/getting-started/installation) (for `cast`)
- `requests` and `pyyaml` Python packages

```bash
pip install requests pyyaml
```

## Quick Start

1. **Create config file**:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your validator details
```

2. **Test with dry-run**:
```bash
python honeyhunter.py --dry-run
```

3. **Compare current vs recommended**:
```bash
python honeyhunter.py --compare
```

4. **Run for real**:
```bash
python honeyhunter.py
```

## Configuration

Edit `config.yaml`:

```yaml
validator:
  pubkey: "0x..."                    # Your validator pubkey (required)
  private_key_file: "/path/to/key"   # Path to private key file (required)
  staking_pool_address: "0x..."      # For staking pool validators (empty for genesis)
  rpc_url: "https://rpc.berachain-apis.com"

contracts:
  berachef_address: "0xfb81E39E3970076ab2693fA5C45A07Cc724C93c2"

strategy:
  efficiency_threshold: 0.5          # Include vaults within 50% of best efficiency

filters:
  min_tvl_usd: 20000                 # Minimum vault TVL
  min_incentive_runway_hours: 3      # Minimum hours of incentives remaining
  min_usd_per_bgt: 0                 # Minimum efficiency (0 = just require incentives exist)
  exclude_protocols: []              # Blacklist protocols by name
  exclude_vaults: []                 # Blacklist vault addresses

limits:
  max_single_vault_pct: 3000         # Max 30% per vault (BeraChef hard limit)
  max_protocol_pct: 5000             # Max 50% to one protocol

execution:
  dry_run: false
  block_buffer: 100                  # Extra blocks added to start_block
  min_change_threshold: 500          # Only update if allocation changes by >5%

api:
  max_retries: 3                     # Retry failed API requests
  retry_delay_seconds: 5             # Base delay (uses exponential backoff)
  timeout_seconds: 30
```

## Allocation Algorithm

The optimizer uses efficiency-weighted allocation:

1. **Sort** vaults by `usdPerBgt` (descending)
2. **Filter** to vaults within `efficiency_threshold` of the best
   - Example: top vault is $0.60/BGT → include vaults ≥$0.30/BGT at 50% threshold
3. **Ensure** at least 4 vaults (required to satisfy 30% cap)
4. **Weight** allocation proportionally to each vault's efficiency
5. **Cap** at 30% per vault, redistribute excess to uncapped vaults
6. **Cap** at 50% per protocol (skipped if all vaults are same protocol)
7. **Normalize** to exactly 10000 basis points (100%)

### Example

| Vault | usdPerBgt | Raw Weight | After 30% Cap | Final |
|-------|-----------|------------|---------------|-------|
| A     | $0.60     | 35%        | 30%           | 30%   |
| B     | $0.50     | 29%        | 29%           | 30%   |
| C     | $0.40     | 23%        | 23%           | 24%   |
| D     | $0.35     | 13%        | 18%           | 16%   |
| E     | $0.20     | --         | --            | excluded (below 50% threshold) |

**Note**: Allocations can be any size - there's no minimum. Total must always equal exactly 100%.

## CLI Options

```bash
python honeyhunter.py --help           # Show help
python honeyhunter.py --dry-run        # Show what would be done
python honeyhunter.py --compare        # Compare current vs recommended
python honeyhunter.py --config FILE    # Use custom config file
python honeyhunter.py -v               # Verbose output (debug logging)
python honeyhunter.py --init           # Create default config file
```

## Automation

### Cron (every 2 hours)

```bash
0 */2 * * * cd /path/to/honeyhunter && /usr/bin/python3 honeyhunter.py >> honeyhunter.log 2>&1
```

### Systemd Timer

Create `/etc/systemd/system/honeyhunter.service`:
```ini
[Unit]
Description=Berachain Reward Allocation Optimizer
After=network.target

[Service]
Type=oneshot
User=your-user
WorkingDirectory=/path/to/honeyhunter
ExecStart=/usr/bin/python3 honeyhunter.py
StandardOutput=append:/path/to/honeyhunter/honeyhunter.log
StandardError=append:/path/to/honeyhunter/honeyhunter.log
```

Create `/etc/systemd/system/honeyhunter.timer`:
```ini
[Unit]
Description=Run Berachain Optimizer every 2 hours

[Timer]
OnCalendar=*:00/2:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now honeyhunter.timer
```

## Example Output

```
2025-01-08 18:30:00 [INFO] ============================================================
2025-01-08 18:30:00 [INFO] HoneyHunter - Berachain Reward Allocation Optimizer
2025-01-08 18:30:00 [INFO] Started at 2025-01-08T18:30:00+00:00
2025-01-08 18:30:00 [INFO] ============================================================
2025-01-08 18:30:01 [INFO] Fetching vault data from Furthermore API...
2025-01-08 18:30:02 [INFO] Fetched 157 valid vaults
2025-01-08 18:30:02 [INFO] Filtered to 42 eligible vaults
2025-01-08 18:30:02 [INFO] Best efficiency: $0.6240/BGT
2025-01-08 18:30:02 [INFO] Efficiency threshold: $0.3120/BGT (50% of best)
2025-01-08 18:30:02 [INFO] Candidates within threshold: 8
2025-01-08 18:30:02 [INFO] Final allocation (5 vaults):
2025-01-08 18:30:02 [INFO]   Kodiak WBTC-WETH: $0.6240/BGT, TVL $5.59M, Runway 24.0h -> 30.0%
2025-01-08 18:30:02 [INFO]   Infrared iBERA: $0.5210/BGT, TVL $47.90M, Runway 72.0h -> 26.0%
2025-01-08 18:30:02 [INFO]   BEX HONEY-WBERA: $0.4980/BGT, TVL $12.30M, Runway 24.0h -> 22.0%
2025-01-08 18:30:02 [INFO]   Kodiak HONEY-USDC: $0.3500/BGT, TVL $8.10M, Runway 36.0h -> 14.0%
2025-01-08 18:30:02 [INFO]   Dolomite dHONEY: $0.3200/BGT, TVL $3.20M, Runway 48.0h -> 8.0%
2025-01-08 18:30:02 [INFO] Total allocation: 100.0%
2025-01-08 18:30:02 [INFO] Current allocation (4 vaults):
2025-01-08 18:30:02 [INFO]   0x1234...: 25.0%
2025-01-08 18:30:02 [INFO]   0x5678...: 25.0%
2025-01-08 18:30:02 [INFO]   0x9abc...: 25.0%
2025-01-08 18:30:02 [INFO]   0xdef0...: 25.0%
2025-01-08 18:30:02 [INFO] Allocation change of 2500 basis points exceeds threshold of 500
2025-01-08 18:30:03 [INFO] Current block: 15500000, delay: 8640, buffer: 100
2025-01-08 18:30:03 [INFO] Start block: 15508740
2025-01-08 18:30:03 [INFO] Executing transaction...
2025-01-08 18:30:08 [INFO] Transaction submitted: 0xabc123...
2025-01-08 18:30:08 [INFO] Waiting for transaction confirmation...
2025-01-08 18:30:15 [INFO] Transaction confirmed successfully!
2025-01-08 18:30:15 [INFO] Success! Transaction: 0xabc123...
```

## The usdPerBgt Metric

`usdPerBgt` represents the USD value of incentives a vault pays per BGT directed to it:

- **Higher is better** for short-term yield
- Calculated by Furthermore based on active incentives and BGT emissions
- Changes frequently as incentives are added/depleted
- This is what Furthermore uses to rank validators by "Rate Per BGT"

## Reliability Features

- **API retry with backoff** - Retries failed API requests (3 attempts, exponential backoff)
- **Transaction confirmation** - Waits for tx receipt and checks success/revert status
- **Queued allocation detection** - Warns if overwriting a pending allocation
- **Duplicate vault filtering** - Handles duplicate addresses from API
- **Data cleansing** - Clamps negative runway values to 0
- **Decimal math** - Uses Python Decimal for precise allocation calculations

## Safety Features

- **Whitelist verification** - Only uses BeraChef whitelisted vaults
- **Active incentives required** - Skips vaults with no incentive value
- **Runway check** - Skips vaults with incentives expiring within threshold
- **Change threshold** - Avoids unnecessary transactions for minor changes
- **Protocol cap bypass** - If all vaults are same protocol, cap is skipped (can't enforce)
- **Guaranteed 100%** - Robust normalization ensures total always equals exactly 10000 bp
- **Dry-run mode** - Test without executing
- **Logging** - Full audit trail in `honeyhunter.log` (UTC timestamps)

## Troubleshooting

### "No validator pubkey configured"
Edit `config.yaml` and add your validator pubkey.

### "No eligible vaults after filtering"
- Lower `min_tvl_usd` or `min_incentive_runway_hours`
- Check if Furthermore API is returning data

### "Only N eligible vaults available, need at least 4"
Not enough vaults pass your filters. Lower thresholds or check API data.

### Transaction fails
- Check your private key file path exists and is readable
- Ensure wallet has `REWARDS_ALLOCATION_MANAGER_ROLE` (for staking pool validators)
- Verify RPC URL is accessible

### "Transaction REVERTED"
- Allocation may violate BeraChef rules (>30% per vault, etc.)
- Check if another allocation was queued between check and submit

### API timeout/connection errors
The script retries automatically. If persistent, check:
- Network connectivity
- Furthermore API status at https://furthermore.app

## Files

```
honeyhunter/
├── honeyhunter.py         # Main script
├── config.yaml.example  # Example configuration (copy to config.yaml)
├── config.yaml          # Your configuration (git-ignored)
├── honeyhunter.log        # Execution log (created on first run, git-ignored)
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## License

MIT
