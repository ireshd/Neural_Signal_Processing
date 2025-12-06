# Examples Directory

This directory contains practical examples demonstrating how to use the Neural Signal DSP modules.

## Available Examples

### 1. `csv_export_example.py`
**Focus:** Signal Generation & Data Export

Demonstrates the neural signal generator with emphasis on CSV export capabilities.

**What it shows:**
- Basic signal generation and export
- Multi-unit recording (3 neurons)
- Different spike waveform types comparison
- Batch processing (multiple trials)

**Run:**
```bash
python examples/csv_export_example.py
```

**Outputs:**
- Multiple CSV files in `data/outputs/`
- Signal data, spike times, summaries

---

### 2. `adc_example.py`
**Focus:** ADC Simulation & Performance Analysis

Demonstrates the ADC simulator with various configurations and scenarios.

**What it shows:**
- Basic ADC conversion (12-bit)
- Resolution comparison (8, 10, 12, 14, 16 bits)
- Timing jitter effects
- Signal saturation behavior
- ADC noise impact
- Realistic neural recording setup (Intan-style)

**Run:**
```bash
python examples/adc_example.py
```

**Outputs:**
- CSV files for each example
- Performance metrics (SNR, ENOB)
- Comparison data

---

### 3. `signal_to_adc_pipeline.py` ⭐
**Focus:** Complete End-to-End Pipeline

Demonstrates the full pipeline from signal generation through ADC conversion.

**What it shows:**
- Complete signal → ADC workflow
- High-resolution analog signal generation
- ADC sampling and digitization
- Performance analysis
- Comprehensive visualization
- Complete data export

**Run:**
```bash
python examples/signal_to_adc_pipeline.py
```

**Outputs:**
- Both analog and digital CSV files
- Complete pipeline visualization
- Performance summary

**This is the recommended starting point for understanding the complete system!**

---

## Quick Start

### Run All Examples

```bash
# Signal generation examples
python examples/csv_export_example.py

# ADC simulation examples
python examples/adc_example.py

# Complete pipeline
python examples/signal_to_adc_pipeline.py
```

### Using Docker

```bash
# Signal examples
docker-compose run neural-dsp python examples/csv_export_example.py

# ADC examples
docker-compose run neural-dsp python examples/adc_example.py

# Pipeline
docker-compose run neural-dsp python examples/signal_to_adc_pipeline.py
```

---

## Output Files

All examples save their outputs to `data/outputs/` directory:

### From `csv_export_example.py`:
- `example_basic_*.csv` - Basic signal data
- `example_multiunit_*.csv` - Multi-unit recording
- `example_unit*_spikes.csv` - Individual unit data
- `waveform_*.csv` - Spike waveform templates
- `trial_*_*.csv` - Batch processing results

### From `adc_example.py`:
- `basic_adc_*.csv` - Basic ADC conversion
- `realistic_neural_recording_*.csv` - Intan-style recording
- Various comparison data files

### From `signal_to_adc_pipeline.py`:
- `pipeline_analog_*.csv` - Original analog signal
- `pipeline_digital_*.csv` - Digitized signal
- `pipeline_complete.png` - Comprehensive visualization

---

## Example Structure

Each example follows this pattern:

```python
# 1. Import modules
from src.signal_gen import NeuralSignalGenerator
from src.adc_sim import ADCSimulator

# 2. Configure parameters
gen = NeuralSignalGenerator(fs=100000, ...)
adc = ADCSimulator(fs=20000, resolution=12, ...)

# 3. Generate/process data
t, signal, spikes, indices = gen.generate_signal(...)
t_dig, sampled, codes, quantized = adc.sample_signal(...)

# 4. Export results
gen.export_to_csv(...)
adc.export_to_csv(...)

# 5. Print summary
print(f"Results saved to data/outputs/")
```

---

## Learning Path

**For beginners:**
1. Start with `signal_to_adc_pipeline.py` - See the complete picture
2. Then explore `csv_export_example.py` - Learn signal generation
3. Finally `adc_example.py` - Deep dive into ADC behavior

**For advanced users:**
- Use these as templates for your own analysis
- Modify parameters to match your specific hardware
- Integrate with your analysis pipelines

---

## Tips

### Customizing Examples

All parameters can be easily modified:

```python
# Change signal characteristics
gen = NeuralSignalGenerator(
    fs=100000,
    noise_amplitude=0.1,  # Increase noise
    drift_amplitude=0.5,  # Stronger drift
    drift_freq=2.0        # Faster drift
)

# Change ADC configuration
adc = ADCSimulator(
    fs=30000,       # Higher sampling rate
    resolution=14,  # 14-bit instead of 12
    vref_pos=2.5,   # Wider voltage range
    enable_jitter=True,
    jitter_std=2e-6 # More jitter
)
```

### Performance Analysis

Compare different configurations:

```python
configs = [
    {'resolution': 12, 'fs': 20000},
    {'resolution': 14, 'fs': 20000},
    {'resolution': 16, 'fs': 30000}
]

for config in configs:
    adc = ADCSimulator(**config)
    # Test and compare...
```

### Batch Processing

Generate multiple trials for statistics:

```python
results = []
for trial in range(100):
    t, signal, spikes, indices = gen.generate_signal(seed=trial)
    results.append({'spikes': len(spikes), 'rate': len(spikes)/duration})

# Analyze statistics
import pandas as pd
df = pd.DataFrame(results)
print(df.describe())
```

---

## Need Help?

- **Documentation:** See `ADC_DOCUMENTATION.md` and `CSV_OUTPUT_FORMAT.md`
- **Quick Start:** See `QUICK_START.md` for basic usage
- **Module Details:** Check inline docstrings in `src/signal_gen.py` and `src/adc_sim.py`

---

## Contributing Examples

Want to add your own example?

1. Create a new file in this directory
2. Follow the existing pattern
3. Include clear comments and docstrings
4. Add entry to this README
5. Test with Docker

**Good example topics:**
- Specific hardware configurations
- Analysis workflows
- Integration with other tools
- Performance optimization
- Real dataset validation

---

**Happy experimenting! 🧪**

