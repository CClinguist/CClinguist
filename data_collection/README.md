# CClinguist Data Collection Tool

This is a toolset for collecting network congestion control algorithm performance data, primarily used for testing different TCP congestion control algorithms under various network conditions.

## Tool Overview

The tool simulates different network conditions (latency, bandwidth) to test TCP congestion control algorithm performance and collects detailed network traffic data for subsequent analysis.

## Main Features

- **Network Condition Simulation**: Uses Mahimahi to simulate different latency and bandwidth conditions
- **TCP Congestion Control Testing**: Supports multiple TCP congestion control algorithms (BBR, Cubic, Reno, etc.)
- **Data Collection**: Automatically collects network traffic data and saves in CSV and PCAP formats
- **Batch Testing**: Supports batch testing with various network parameter combinations
- **Real Website Testing**: Supports performance testing on real websites

## Installation Dependencies

Before running the tool, please install the necessary dependencies:

```bash
# Install system dependencies
sudo ./dependency.sh

# Install Mahimahi
# Please refer to the official Mahimahi documentation for installation

```

## Usage

### Basic Usage

1. **Configure target parameters**:
   Edit the following parameters in the `run.py` file:

```python
# Target IP address
targetIP = "your_target_ip"

# Used for real detection
remote_machine = ""

# TCP congestion control algorithm list
tcpCC_List = ["dctcp"]  # Options: ["bbr", "cubic"...]  ccas in linux
```

2. **Run data collection**:
```bash
python3 run.py
```

### Parameter Configuration

#### Latency Parameter Adjustment

Modify the following parameters in the `run_launch` function in `run.py`:

```python
# Target latency range (milliseconds)
targetDelay_List = list(range(40, 220, 20))  # From 40ms to 200ms, step 20ms

# Example: Modify to a smaller latency range
targetDelay_List = list(range(50, 150, 25))  # From 50ms to 125ms, step 25ms
```

#### Bandwidth Parameter Adjustment

Modify the following parameters in the `run_launch` function in `run.py`:

```python
# Target bandwidth range (Kbps)
targetBW_List = list(range(400, 1100, 200))  # From 400Kbps to 900Kbps, step 200Kbps

# Example: Modify to a higher bandwidth range
targetBW_List = list(range(1000, 5000, 500))  # From 1000Kbps to 4500Kbps, step 500Kbps
```

### Advanced Configuration

#### 1. Custom Network Conditions

Modify network simulation parameters in `simnet.sh`:

```bash
# Buffer size calculation (multiple of BDP)
buffBDP=1

# AQM algorithm
aqm=droptail  # Options: droptail, red, codel
```

#### 2. Custom Test Targets

For real website testing, modify target configuration in `run.py`:

```python
targetLink = "https://www.example.com"
targetIP = "1.2.3.4"

# Enable in run_launch function
run_launch(remote_machine, targetLink, targetIP, base_bw, tcpCC_List, i, 1)
```

#### 3. Batch Testing Configuration

```python
# Multiple rounds of testing
for i in range(3):  # Run 3 rounds of tests
    run_launch(remote_machine, targetLink, targetIP, base_bw, tcpCC_List, i, 1)
```
