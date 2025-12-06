# CSV Output Format Documentation

The Neural Signal Generator exports data in multiple CSV files for easy analysis in spreadsheet software, Python, MATLAB, R, or other data analysis tools.

## Exported Files

When you run the signal generator, it creates the following CSV files in the `data/outputs/` directory:

### 1. Signal Data (`neural_signal_data.csv`)

Complete time-series data of the generated neural signal.

**Format:**
```
time_s,amplitude,spike_marker
0.000000,0.123456,0.0
0.000050,0.234567,0.0
0.000100,0.345678,1.0
...
```

**Columns:**
- `time_s`: Time in seconds
- `amplitude`: Signal amplitude (normalized voltage)
- `spike_marker`: Binary marker (1.0 = spike present, 0.0 = no spike)

**Use cases:**
- Time-domain analysis
- Custom filtering
- Visualization in Excel/Python/MATLAB
- Machine learning training data

---

### 2. Spike Times (`neural_signal_spike_times.csv`)

Precise timing of detected/generated spikes.

**Format:**
```
spike_time_s,spike_index
0.052341,1047
0.089123,1782
0.134567,2691
...
```

**Columns:**
- `spike_time_s`: Spike occurrence time in seconds
- `spike_index`: Sample index of spike (for array indexing)

**Use cases:**
- Inter-spike interval (ISI) analysis
- Firing rate calculations
- Raster plots
- Spike train statistics

---

### 3. Spike Waveform Template (`spike_waveform_template.csv`)

The template waveform used to generate spikes.

**Format:**
```
time_s,amplitude
0.000000,0.001234
0.000050,-0.234567
0.000100,-0.789012
...
```

**Columns:**
- `time_s`: Time relative to spike onset (seconds)
- `amplitude`: Normalized amplitude

**Use cases:**
- Waveform analysis
- Template matching
- Spike sorting validation
- Feature extraction (peak amplitude, width, etc.)

---

### 4. Signal Summary (`neural_signal_summary.csv`)

Metadata and statistics about the generated signal.

**Format:**
```
parameter,value,unit
duration,1.0,s
sampling_frequency,20000.0,Hz
num_samples,20000,count
num_spikes,32,count
firing_rate,32.00,Hz
noise_amplitude,0.05,V
drift_amplitude,0.2,V
drift_frequency,1.0,Hz
signal_mean,0.012345,V
signal_std,0.156789,V
signal_min,-1.234567,V
signal_max,0.987654,V
```

**Parameters:**
- `duration`: Total signal duration
- `sampling_frequency`: Sampling rate (fs)
- `num_samples`: Total number of samples
- `num_spikes`: Number of generated spikes
- `firing_rate`: Average spikes per second
- `noise_amplitude`: Noise standard deviation
- `drift_amplitude`: Low-frequency drift amplitude
- `drift_frequency`: Drift component frequency
- `signal_mean`: Mean signal value
- `signal_std`: Signal standard deviation
- `signal_min`: Minimum signal value
- `signal_max`: Maximum signal value

**Use cases:**
- Experiment documentation
- Parameter tracking
- Batch analysis metadata
- Quality control checks

---

## Loading CSV Files

### Python (pandas)

```python
import pandas as pd

# Load signal data
signal_df = pd.read_csv('data/outputs/neural_signal_data.csv')
time = signal_df['time_s'].values
amplitude = signal_df['amplitude'].values

# Load spike times
spikes_df = pd.read_csv('data/outputs/neural_signal_spike_times.csv')
spike_times = spikes_df['spike_time_s'].values

# Load summary
summary_df = pd.read_csv('data/outputs/neural_signal_summary.csv')
fs = summary_df[summary_df['parameter'] == 'sampling_frequency']['value'].values[0]
```

### Python (numpy)

```python
import numpy as np

# Load signal data
data = np.loadtxt('data/outputs/neural_signal_data.csv', 
                  delimiter=',', skiprows=1)
time = data[:, 0]
amplitude = data[:, 1]
spike_marker = data[:, 2]

# Load spike times
spikes = np.loadtxt('data/outputs/neural_signal_spike_times.csv',
                    delimiter=',', skiprows=1)
spike_times = spikes[:, 0]
```

### MATLAB

```matlab
% Load signal data
data = readtable('data/outputs/neural_signal_data.csv');
time = data.time_s;
amplitude = data.amplitude;

% Load spike times
spikes = readtable('data/outputs/neural_signal_spike_times.csv');
spike_times = spikes.spike_time_s;

% Load summary
summary = readtable('data/outputs/neural_signal_summary.csv');
```

### R

```r
# Load signal data
signal_data <- read.csv('data/outputs/neural_signal_data.csv')
time <- signal_data$time_s
amplitude <- signal_data$amplitude

# Load spike times
spikes <- read.csv('data/outputs/neural_signal_spike_times.csv')
spike_times <- spikes$spike_time_s

# Load summary
summary <- read.csv('data/outputs/neural_signal_summary.csv')
```

### Excel / Google Sheets

Simply open the CSV files directly. They will automatically parse into columns.

---

## Programmatic Export

You can generate and export data programmatically:

```python
from src.signal_gen import NeuralSignalGenerator

# Create generator
gen = NeuralSignalGenerator(fs=20000)

# Generate signal
t, signal, spike_times, spike_indices = gen.generate_signal(
    duration=1.0,
    firing_rate=30.0,
    spike_type='biphasic',
    seed=42
)

# Export to CSV
exported_files = gen.export_to_csv(
    t, signal, spike_times, spike_indices,
    output_dir='data/outputs',
    prefix='my_experiment'
)

# Export waveform template
waveform_path = gen.export_waveform_to_csv(
    spike_type='biphasic',
    output_dir='data/outputs',
    filename='my_waveform.csv'
)
```

See `examples/csv_export_example.py` for more advanced usage.

---

## File Naming Convention

Default names:
- `neural_signal_data.csv`
- `neural_signal_spike_times.csv`
- `neural_signal_summary.csv`
- `spike_waveform_template.csv`

Custom prefixes (using `prefix` parameter):
- `{prefix}_data.csv`
- `{prefix}_spike_times.csv`
- `{prefix}_summary.csv`

---

## Data Integrity

All CSV files include:
- ✓ Header rows with column names
- ✓ Consistent decimal precision
- ✓ Standard comma delimiters
- ✓ UTF-8 encoding
- ✓ Unix/Windows compatible line endings

---

## Tips

1. **Large Files**: For long recordings (>60 seconds at 20 kHz), signal data files can be >10 MB. Consider:
   - Downsampling for visualization
   - Using binary formats (HDF5, NPY) for very large datasets
   - Processing in chunks

2. **Batch Processing**: Use the `prefix` parameter to organize multiple trials:
   ```python
   for trial in range(10):
       exported_files = gen.export_to_csv(..., prefix=f'trial_{trial:02d}')
   ```

3. **Time Alignment**: All timestamps are in seconds and share the same time base across files.

4. **Missing Spikes**: If no spikes are generated, `spike_times.csv` will still be created with just the header row.

