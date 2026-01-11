# 🍯 HoneyHunter

Automated BGT reward allocation optimizer for Berachain validators. Hunts for the highest-yielding vaults based on `usdPerBgt` efficiency.

## How It Works

1. **Fetches** vault data from the Furthermore API (with retry logic)
2. **Filters** vaults by: whitelisted, has active incentives, minimum incentive value, runway, optional TVL/blacklists
3. **Sorts** by `usdPerBgt` - USD value of incentives per BGT directed
4. **Selects** top N vaults (configurable, default 4)
5. **Allocates** using greedy (30/30/30/10) or proportional weighting
6. **Executes** via `cast send` with transaction confirmation

## Prerequisites

- Python 3.11+
- [Foundry](https://book.getfoundry.sh/getting-started/installation) (for `cast`)

## Installation

```bash
# Clone the repo
git clone https://github.com/RhinoStake/HoneyHunter.git
cd honeyhunter

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Create config file**:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your validator details
```

2. **Test with dry-run**:
```bash
source venv/bin/activate  # If not already activated
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
  berachef_address: "0xdf960E8F3F19C481dDE769edEDD439ea1a63426a"

strategy:
  max_vaults: 4                      # How many vaults to allocate to (4-10)
  allocation_mode: "greedy"          # "greedy" or "proportional"
  efficiency_threshold: 0.85         # For proportional mode only

filters:
  min_tvl_usd: 0                     # Minimum vault TVL (0 = no minimum)
  min_incentive_runway_hours: 3      # Minimum hours of incentives remaining
  min_incentive_value: 0             # Minimum incentive value in $thousands (0 = any)
  min_usd_per_bgt: 0                 # Minimum efficiency (0 = any)
  exclude_protocols: []              # Blacklist protocols by name
  exclude_vaults: []                 # Blacklist vault addresses

limits:
  max_single_vault_pct: 3000         # Max 30% per vault (BeraChef limit)
  max_protocol_pct: 5000             # Max 50% to one protocol (proportional mode)

execution:
  dry_run: false
  block_buffer: 100                  # Extra blocks added to start_block
  min_change_threshold: 500          # Only update if allocation changes by >5%

api:
  max_retries: 3
  retry_delay_seconds: 5
  timeout_seconds: 30

healthchecks:
  enabled: false
  id: ""                             # Your healthchecks.io UUID
```

## Allocation Modes

### Greedy Mode (Default) - Maximum Concentration

Allocates 30% (the BeraChef maximum) to each vault in order of USD/BGT until the budget is exhausted. The last vault gets the remainder.

**For 4 vaults:** 30% / 30% / 30% / 10%

This maximizes your allocation to the highest-yielding vaults.

```yaml
strategy:
  max_vaults: 4
  allocation_mode: "greedy"
```

### Proportional Mode - Efficiency-Weighted Distribution

Weights allocation proportionally to each vault's USD/BGT efficiency, respecting the 30% per-vault cap and optional protocol caps.

Uses `efficiency_threshold` to filter candidates first (e.g., 0.85 = only vaults within 85% of the best).

```yaml
strategy:
  max_vaults: 8
  allocation_mode: "proportional"
  efficiency_threshold: 0.85
```

### Example Comparison

Given vaults with USD/BGT of $0.60, $0.50, $0.40, $0.35:

| Mode | Vault A ($0.60) | Vault B ($0.50) | Vault C ($0.40) | Vault D ($0.35) |
|------|-----------------|-----------------|-----------------|-----------------|
| Greedy | 30% | 30% | 30% | 10% |
| Proportional | 30% | 27% | 22% | 21% |

Greedy concentrates 90% in top 3; Proportional spreads more evenly.

## Strategy Presets

### Aggressive (Max Returns)
```yaml
strategy:
  max_vaults: 4
  allocation_mode: "greedy"

filters:
  min_tvl_usd: 0
  min_incentive_runway_hours: 3
```

### Conservative (Risk-Adjusted)
```yaml
strategy:
  max_vaults: 8
  allocation_mode: "proportional"
  efficiency_threshold: 0.85

filters:
  min_tvl_usd: 50000
  min_incentive_runway_hours: 24
```

## CLI Options

```bash
python honeyhunter.py --help           # Show help
python honeyhunter.py --dry-run        # Show what would be done
python honeyhunter.py --compare        # Compare current vs recommended
python honeyhunter.py --config FILE    # Use custom config file
python honeyhunter.py -v               # Verbose output (debug logging)
```

## Automation

### Cron (every 2 hours)

```bash
0 */2 * * * cd /path/to/honeyhunter && ./venv/bin/python honeyhunter.py
```

The script logs to `honeyhunter.log` automatically.

### Systemd Timer

Create `/etc/systemd/system/honeyhunter.service`:
```ini
[Unit]
Description=HoneyHunter - Berachain Reward Allocation Optimizer
After=network.target

[Service]
Type=oneshot
User=your-user
WorkingDirectory=/path/to/honeyhunter
ExecStart=/path/to/honeyhunter/venv/bin/python honeyhunter.py
```

Create `/etc/systemd/system/honeyhunter.timer`:
```ini
[Unit]
Description=Run HoneyHunter every 2 hours

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
2025-01-08 18:30:02 [INFO] Allocation mode: greedy
2025-01-08 18:30:02 [INFO] Max vaults: 4
2025-01-08 18:30:02 [INFO] Selected 4 vaults:
2025-01-08 18:30:02 [INFO]   Kodiak WBTC-WETH: $0.6240/BGT, runway 24.0h
2025-01-08 18:30:02 [INFO]   Infrared iBERA: $0.5210/BGT, runway 72.0h
2025-01-08 18:30:02 [INFO]   BEX HONEY-WBERA: $0.4980/BGT, runway 24.0h
2025-01-08 18:30:02 [INFO]   Kodiak HONEY-USDC: $0.3500/BGT, runway 36.0h
2025-01-08 18:30:02 [INFO] Greedy allocation result:
2025-01-08 18:30:02 [INFO]   0x1234...: 30%
2025-01-08 18:30:02 [INFO]   0x5678...: 30%
2025-01-08 18:30:02 [INFO]   0x9abc...: 30%
2025-01-08 18:30:02 [INFO]   0xdef0...: 10%
2025-01-08 18:30:02 [INFO] Current block: 15500000, delay: 8640, buffer: 100
2025-01-08 18:30:02 [INFO] Start block: 15508740
2025-01-08 18:30:02 [INFO] Executing transaction...
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
- **Queued allocation detection** - Errors if a pending allocation exists (prevents wasted gas)
- **Duplicate vault filtering** - Handles duplicate addresses from API
- **Data cleansing** - Clamps negative runway values to 0
- **Decimal math** - Uses Python Decimal for precise allocation calculations
- **Healthchecks.io integration** - Optional cron monitoring (set `healthchecks.enabled: true`)

## Safety Features

- **Whitelist verification** - Only uses BeraChef whitelisted vaults
- **Active incentives required** - Skips vaults with no incentive value
- **Minimum incentive value** - Optional threshold for total active incentive USD
- **Runway check** - Skips vaults with incentives expiring within threshold
- **Change threshold** - Avoids unnecessary transactions for minor changes
- **Guaranteed 100%** - Robust normalization ensures total always equals exactly 10000 bp
- **Dry-run mode** - Test without executing
- **Logging** - Full audit trail in `honeyhunter.log` (UTC timestamps)

## Troubleshooting

### "No validator pubkey configured"
Edit `config.yaml` and add your validator pubkey.

### "No eligible vaults after filtering"
- Lower `min_tvl_usd`, `min_incentive_value`, or `min_incentive_runway_hours`
- Check if Furthermore API is returning data

### "Only N eligible vaults available, need at least 4"
Not enough vaults pass your filters. Lower thresholds or check API data.

### Transaction fails
- Check your private key file path exists and is readable
- Ensure wallet has `REWARDS_ALLOCATION_MANAGER_ROLE` (for staking pool validators)
- Verify RPC URL is accessible

### "Transaction REVERTED"
- Allocation may violate BeraChef rules (>30% per vault, >10 vaults, etc.)
- Check if another allocation was queued between check and submit

### API timeout/connection errors
The script retries automatically. If persistent, check:
- Network connectivity
- Furthermore API status at https://furthermore.app

## Files

```
honeyhunter/
├── honeyhunter.py       # Main script
├── config.yaml.example  # Example configuration (copy to config.yaml)
├── config.yaml          # Your configuration (git-ignored)
├── honeyhunter.log      # Execution log (created on first run, git-ignored)
├── requirements.txt     # Python dependencies
├── venv/                # Virtual environment (git-ignored)
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT
