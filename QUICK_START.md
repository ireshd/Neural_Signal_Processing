# Quick Start Guide

Get up and running with the Neural Signal DSP Pipeline in under 5 minutes!

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (usually included with Docker Desktop)

## Step 1: Build the Docker Image

Open a terminal/command prompt in the project directory and run:

**Windows (PowerShell/CMD):**
```bash
docker-compose build
```

**Linux/Mac:**
```bash
docker-compose build
```

This downloads Python, installs dependencies, and prepares the environment (~2-3 minutes).

## Step 2: Generate Your First Neural Signal

Run the signal generator:

```bash
docker-compose run neural-dsp python src/signal_gen.py
```

**What happens:**
1. Generates 1 second of synthetic neural signal (20,000 samples)
2. Injects ~30 spikes with realistic waveforms
3. Saves plot to `data/outputs/neural_signal_demo.png`
4. Exports 4 CSV files with complete data

## Step 3: View Your Results

### Plots

Open the generated plot:
- **Location:** `data/outputs/neural_signal_demo.png`
- Shows: Full signal, zoomed view, and spike waveform template

### CSV Data

Open CSV files in Excel, Python, MATLAB, or any spreadsheet software:

**Files created:**
1. `data/outputs/neural_signal_data.csv` - Complete signal time-series
2. `data/outputs/neural_signal_spike_times.csv` - Spike timing data
3. `data/outputs/spike_waveform_template.csv` - Spike template
4. `data/outputs/neural_signal_summary.csv` - Statistics summary

See `CSV_OUTPUT_FORMAT.md` for detailed format information.

## Step 4: Run the ADC Simulator

Try the ADC simulation:

```bash
docker-compose run neural-dsp python src/adc_sim.py
```

**What happens:**
1. Generates neural signal at 100 kHz (analog)
2. Samples at 20 kHz with 8-bit, 12-bit, and 16-bit ADCs
3. Compares quantization effects
4. Saves plots and CSV data

## Step 5: Run Advanced Examples

Try the CSV export examples:

```bash
docker-compose run neural-dsp python examples/csv_export_example.py
```

This generates:
- Basic single-unit recording
- Multi-unit recording (3 neurons)
- Different spike waveform types
- Batch processing (5 trials)

Try the ADC examples:

```bash
docker-compose run neural-dsp python examples/adc_example.py
```

This demonstrates:
- Resolution comparison (8-16 bits)
- Timing jitter effects
- Signal saturation
- ADC noise impact
- Realistic neural recording setup

All outputs are saved to `data/outputs/` directory.

## Step 6: Analyze Your Data

### Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the signal
df = pd.read_csv('data/outputs/neural_signal_data.csv')

# Plot it
plt.figure(figsize=(12, 4))
plt.plot(df['time_s'], df['amplitude'])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Neural Signal')
plt.show()
```

### MATLAB

```matlab
% Load the signal
data = readtable('data/outputs/neural_signal_data.csv');

% Plot it
figure;
plot(data.time_s, data.amplitude);
xlabel('Time (s)');
ylabel('Amplitude');
title('Neural Signal');
```

### R

```r
# Load the signal
data <- read.csv('data/outputs/neural_signal_data.csv')

# Plot it
plot(data$time_s, data$amplitude, type='l',
     xlab='Time (s)', ylab='Amplitude',
     main='Neural Signal')
```

### Excel

Simply open any CSV file with Excel or Google Sheets!

## Customizing Parameters

Edit `src/signal_gen.py` or create your own script:

```python
from src.signal_gen import NeuralSignalGenerator

# Create custom generator
gen = NeuralSignalGenerator(
    fs=20000,              # Sampling rate (Hz)
    noise_amplitude=0.1,   # More noise
    drift_amplitude=0.3,   # Stronger drift
    drift_freq=0.5         # Slower drift (0.5 Hz)
)

# Generate longer signal with more spikes
t, signal, spike_times, spike_indices = gen.generate_signal(
    duration=5.0,          # 5 seconds
    firing_rate=50.0,      # 50 Hz firing rate
    spike_type='triphasic',# Different waveform
    seed=42
)

# Export everything
exported = gen.export_to_csv(t, signal, spike_times, spike_indices,
                            output_dir='data/outputs',
                            prefix='custom_signal')
```

## Common Commands

**Generate neural signal:**
```bash
docker-compose run neural-dsp python src/signal_gen.py
```

**Simulate ADC:**
```bash
docker-compose run neural-dsp python src/adc_sim.py
```

**Run signal examples:**
```bash
docker-compose run neural-dsp python examples/csv_export_example.py
```

**Run ADC examples:**
```bash
docker-compose run neural-dsp python examples/adc_example.py
```

**Interactive Python shell:**
```bash
docker-compose run neural-dsp python
```

**Shell access (for debugging):**
```bash
docker-compose run neural-dsp /bin/bash
```

**Clean up:**
```bash
docker-compose down
```

## Troubleshooting

### "docker-compose: command not found"

Try `docker compose` (without hyphen) instead:
```bash
docker compose run neural-dsp python src/signal_gen.py
```

### Permission errors (Linux)

Add your user ID:
```bash
docker-compose run --user $(id -u):$(id -g) neural-dsp python src/signal_gen.py
```

### Output directory doesn't exist

Create it first:
```bash
mkdir -p data/outputs
docker-compose run neural-dsp python src/signal_gen.py
```

### Want to see the plot interactively (not just saved)

The Docker container uses a non-interactive backend by default. To see plots:
1. Save them to files (default behavior) ✓
2. Or run outside Docker: `python src/signal_gen.py`

## Step 7: Run Tests (Optional)

Validate that everything works correctly:

```bash
docker-compose run neural-dsp python testbench/run_all_tests.py
```

This runs 100+ tests covering:
- Signal generation correctness
- ADC conversion accuracy
- Integration pipeline
- Edge cases and error handling

Expected output:
```
✓ ALL TESTS PASSED! System validated and ready for use.
```

## Next Steps

1. **Explore parameters** - Try different firing rates, noise levels, spike types
2. **Batch processing** - Generate multiple trials for statistical analysis
3. **Build the full pipeline** - Implement DMA buffer, DSP filters, spike detection
4. **Custom analysis** - Load CSV files into your favorite analysis tool
5. **Run tests** - Validate functionality with comprehensive test suite

## Need Help?

- See `README.md` for complete documentation
- See `CSV_OUTPUT_FORMAT.md` for CSV format details
- See `README_DOCKER.md` for Docker-specific info
- Check `examples/csv_export_example.py` for code examples

---

**You're all set! 🎉 Start generating neural signals!**

