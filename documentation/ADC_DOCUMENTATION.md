# ADC Simulation Documentation

Complete documentation for the ADC (Analog-to-Digital Converter) simulation module.

## Overview

The `ADCSimulator` class simulates the behavior of real hardware ADCs used in neural recording systems. It models key characteristics including:

- **Quantization** - Converting continuous analog signals to discrete digital codes
- **Sampling** - Capturing signal values at regular time intervals
- **Timing Jitter** - Random variations in sample timing
- **Saturation** - Clipping when signals exceed reference voltages
- **ADC Noise** - Additional noise introduced by the conversion process

## Quick Start

```python
from src.adc_sim import ADCSimulator
from src.signal_gen import NeuralSignalGenerator

# Generate an analog signal
gen = NeuralSignalGenerator(fs=100000)
t, signal, _, _ = gen.generate_signal(duration=1.0)

# Create 12-bit ADC at 20 kHz
adc = ADCSimulator(fs=20000, resolution=12)

# Sample and digitize
t_digital, sampled, codes, quantized = adc.sample_signal(t, signal)

# Get performance metrics
snr = adc.get_snr(sampled, quantized)
print(f"SNR: {snr:.2f} dB")
```

## Class: ADCSimulator

### Initialization Parameters

```python
ADCSimulator(
    fs=20000.0,              # Sampling frequency (Hz)
    resolution=12,           # Bit resolution (8-16 typical)
    vref_pos=1.65,          # Positive reference voltage (V)
    vref_neg=-1.65,         # Negative reference voltage (V)
    enable_jitter=False,    # Enable timing jitter
    jitter_std=1e-6,        # Jitter standard deviation (s)
    enable_saturation=True, # Enable saturation/clipping
    adc_noise_std=0.0       # ADC noise std dev (V)
)
```

#### Parameter Details

**`fs` (float)**: Sampling frequency in Hz
- Typical neural recording: 10-30 kHz
- Higher frequencies capture faster dynamics
- Nyquist theorem: must be ≥ 2× highest signal frequency

**`resolution` (int)**: Number of bits
- 8-bit: 256 levels, fast but low precision
- 12-bit: 4096 levels, good balance (common in embedded)
- 16-bit: 65536 levels, high precision (research systems)

**`vref_pos`, `vref_neg` (float)**: Reference voltages
- Define the measurable voltage range
- Signals outside this range are saturated (clipped)
- Example: ±1.65V (typical for 3.3V systems)

**`enable_jitter` (bool)**: Sampling time jitter
- Models imperfect clock timing
- Degrades effective resolution at high frequencies
- Typical: 0-10 μs standard deviation

**`enable_saturation` (bool)**: Signal clipping
- True: clips signals to [vref_neg, vref_pos]
- False: allows out-of-range values (unrealistic)

**`adc_noise_std` (float)**: ADC noise level
- Models thermal noise, quantization noise
- Typical: 0.1-1 mV for neural ADCs
- Reduces effective SNR

### Key Properties

```python
adc.lsb          # Least Significant Bit voltage (V)
adc.max_code     # Maximum digital code (2^resolution - 1)
adc.voltage_range # Total voltage range (vref_pos - vref_neg)
adc.offset       # Voltage offset (vref_neg)
```

**LSB (Least Significant Bit)**:
```
LSB = voltage_range / (2^resolution - 1)
```

Example: 12-bit ADC with ±1V range
```
LSB = 2.0 / (2^12 - 1) = 0.488 mV
```

### Main Methods

#### 1. `sample_signal()`

Sample an analog signal at the ADC sampling rate.

```python
t_digital, sampled_analog, digital_codes, quantized_voltage = adc.sample_signal(
    time,           # Input time array
    analog_signal,  # Input signal array
    target_fs=None  # Optional: override sampling rate
)
```

**Returns:**
- `t_digital`: Sample times (with jitter if enabled)
- `sampled_analog`: Interpolated analog values at sample times
- `digital_codes`: Integer ADC codes [0, 2^resolution-1]
- `quantized_voltage`: Reconstructed voltage from digital codes

**Process:**
1. Generate sample times based on `fs`
2. Add jitter if enabled
3. Interpolate input signal at sample times
4. Convert to digital codes (with saturation, noise)
5. Convert codes back to quantized voltage

#### 2. `analog_to_digital()`

Convert analog voltage to digital codes.

```python
digital_codes, quantized_voltage = adc.analog_to_digital(analog_signal)
```

This method:
- Clips to voltage range (if saturation enabled)
- Adds ADC noise (if specified)
- Quantizes to discrete levels
- Returns both codes and reconstructed voltage

#### 3. `quantize_only()`

Apply quantization without changing sample rate.

```python
quantized = adc.quantize_only(analog_signal)
```

Useful for seeing quantization effects independently from sampling.

#### 4. `get_snr()`

Calculate Signal-to-Noise Ratio from quantization error.

```python
snr_db = adc.get_snr(original, quantized)
```

**Formula:**
```
SNR = 10 * log10(signal_power / noise_power)
```

where noise is the quantization error.

**Typical values:**
- 8-bit: ~48 dB
- 12-bit: ~72 dB
- 16-bit: ~96 dB

#### 5. `get_enob()`

Calculate Effective Number of Bits.

```python
enob = adc.get_enob(original, quantized)
```

**Formula:**
```
ENOB = (SNR_dB - 1.76) / 6.02
```

ENOB accounts for noise, jitter, and non-idealities that reduce effective resolution.

Example: A 12-bit ADC might have ENOB = 10.5 bits due to noise.

#### 6. `get_statistics()`

Get comprehensive ADC statistics.

```python
stats = adc.get_statistics()
```

Returns dictionary with:
- `resolution_bits`: Nominal bit resolution
- `sampling_rate_hz`: Sampling frequency
- `lsb_voltage`: LSB size
- `num_saturated_samples`: Count of clipped samples
- `saturation_percentage`: % of samples saturated
- And more...

#### 7. `export_to_csv()`

Export ADC data to CSV files.

```python
exported_files = adc.export_to_csv(
    time, analog_signal, digital_codes, quantized_signal,
    output_dir='data/outputs',
    prefix='adc_output'
)
```

Creates 3 CSV files:
1. `{prefix}_data.csv` - Time, analog, codes, quantized
2. `{prefix}_quantization_error.csv` - Time, error
3. `{prefix}_statistics.csv` - Performance metrics

## Key Concepts

### Quantization

Converting continuous voltages to discrete digital codes.

**Quantization Error**: ±0.5 LSB
```
Error = Original - Quantized
Maximum Error = ±LSB/2
```

**Quantization Noise**:
```
σ_q = LSB / √12  (for uniform quantization)
```

### Sampling

Capturing signal values at discrete time points.

**Nyquist-Shannon Theorem**: 
```
fs ≥ 2 × f_max
```

For neural signals (DC-3 kHz), minimum fs = 6 kHz, but 20 kHz is common for headroom.

### Saturation

Clipping when signal exceeds ADC range.

```python
if voltage > vref_pos:
    voltage = vref_pos  # Clip to max
if voltage < vref_neg:
    voltage = vref_neg  # Clip to min
```

**Effects:**
- Distorts signal peaks
- Loss of information
- Can indicate need for gain adjustment

### Timing Jitter

Random variations in sample timing.

```python
actual_time = ideal_time + random_jitter
```

**Effects:**
- Equivalent to phase noise
- Reduces SNR at high frequencies
- More critical for high-resolution ADCs

**SNR degradation from jitter:**
```
SNR_jitter = -20 * log10(2π × f_signal × σ_jitter)
```

## Practical Examples

### Example 1: Basic 12-bit ADC

```python
adc = ADCSimulator(
    fs=20000,
    resolution=12,
    vref_pos=1.0,
    vref_neg=-1.0
)

t, _, codes, quantized = adc.sample_signal(t_analog, signal_analog)

print(f"LSB: {adc.lsb * 1000:.3f} mV")
print(f"Voltage range: ±{adc.vref_pos} V")
print(f"Max code: {adc.max_code}")
```

### Example 2: High-Resolution Research System

```python
# Intan RHD2000-like setup
adc = ADCSimulator(
    fs=20000,
    resolution=16,
    vref_pos=0.005,      # ±5 mV (after amplification)
    vref_neg=-0.005,
    enable_jitter=True,
    jitter_std=0.5e-6,   # 0.5 μs
    adc_noise_std=0.0001 # 0.1 mV
)
```

### Example 3: Compare Resolutions

```python
for resolution in [8, 10, 12, 14, 16]:
    adc = ADCSimulator(fs=20000, resolution=resolution)
    _, sampled, _, quantized = adc.sample_signal(t, signal)
    snr = adc.get_snr(sampled, quantized)
    print(f"{resolution}-bit: LSB={adc.lsb*1000:.3f} mV, SNR={snr:.1f} dB")
```

### Example 4: Check for Saturation

```python
adc = ADCSimulator(fs=20000, resolution=12, enable_saturation=True)
t, sampled, codes, quantized = adc.sample_signal(t_analog, signal_analog)

stats = adc.get_statistics()
if stats['saturation_percentage'] > 1.0:
    print(f"WARNING: {stats['saturation_percentage']:.1f}% of samples saturated!")
    print("Consider: increasing Vref or reducing signal amplitude")
```

## Performance Metrics

### Theoretical SNR

For ideal N-bit ADC with full-scale sinusoid:
```
SNR_ideal = 6.02N + 1.76 dB
```

Examples:
- 8-bit: 49.9 dB
- 12-bit: 74.0 dB
- 16-bit: 98.1 dB

### Effective Resolution (ENOB)

Real ADCs have lower performance than ideal:

```python
enob = adc.get_enob(original, quantized)
resolution_loss = adc.resolution - enob
print(f"Lost {resolution_loss:.1f} bits to noise/jitter")
```

Typical ENOB losses:
- Good ADC: 0.5-1.0 bits
- With jitter/noise: 1.0-2.0 bits
- Poor design: >2.0 bits

## Common Use Cases

### 1. ADC Selection

Compare different ADC configurations:
```python
configs = [
    (8, 0.488),   # 8-bit, cheap
    (12, 0.488),  # 12-bit, standard
    (16, 0.076)   # 16-bit, premium
]

for resolution, lsb_mv in configs:
    # Test with your signal...
    # Check if SNR is sufficient
```

### 2. Range Optimization

Find optimal Vref for your signal:
```python
signal_peak = np.max(np.abs(signal))
margin = 1.2  # 20% headroom

optimal_vref = signal_peak * margin
print(f"Recommended Vref: ±{optimal_vref:.3f} V")
```

### 3. Saturation Detection

Monitor signal integrity:
```python
if stats['saturation_percentage'] > 0:
    print("⚠ Signal clipping detected!")
    print(f"  Clipped: {stats['num_saturated_samples']} samples")
    print(f"  Consider increasing Vref or reducing gain")
```

## CSV Output Format

### File 1: `{prefix}_data.csv`

```csv
time_s,analog_voltage,digital_code,quantized_voltage
0.000000,0.123456,2048,0.122070
0.000050,0.234567,2345,0.234375
...
```

### File 2: `{prefix}_quantization_error.csv`

```csv
time_s,error_voltage
0.000000,0.001386
0.000050,0.000192
...
```

### File 3: `{prefix}_statistics.csv`

```csv
parameter,value,unit
resolution,12,bits
sampling_rate,20000.0,Hz
lsb_voltage,0.000488281,V
snr,72.45,dB
enob,11.76,bits
...
```

## Tips & Best Practices

1. **Choose appropriate resolution**
   - Neural spikes: 12-14 bits usually sufficient
   - High-SNR applications: 16 bits
   - Cost-sensitive: 10 bits minimum

2. **Set proper voltage range**
   - Too high → wasted resolution
   - Too low → saturation
   - Aim for 80-90% of full scale

3. **Monitor saturation**
   - Always check saturation statistics
   - <0.1% is acceptable
   - >1% indicates problems

4. **Consider sampling rate**
   - Nyquist: fs > 2× signal bandwidth
   - Practical: fs = 5-10× bandwidth
   - Neural: 20-30 kHz typical

5. **Account for noise**
   - Signal noise + ADC noise
   - ENOB decreases with noise
   - May need higher resolution than theoretical

## References

- **Intan Technologies**: RHD2000 series datasheets
- **Texas Instruments**: "Understanding Data Converters" (SLAA013)
- **Analog Devices**: ADC selection guides

See `examples/adc_example.py` for complete working examples.

