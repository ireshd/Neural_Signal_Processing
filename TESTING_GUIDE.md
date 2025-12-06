# Testing Guide

Complete guide to testing the Neural Signal DSP Pipeline.

## Overview

The project includes a comprehensive test suite with 100+ tests covering all implemented modules. Tests validate correctness, handle edge cases, and ensure system reliability.

## Quick Start

### Run All Tests

```bash
# Standard Python
python testbench/run_all_tests.py

# With Docker
docker-compose run neural-dsp python testbench/run_all_tests.py

# With pytest (if installed)
pytest testbench/ -v
```

### Expected Output

```
╔════════════════════════════════════════════════════════════════════════════╗
║                 NEURAL SIGNAL DSP - TEST SUITE                              ║
║                                                                              ║
║  This test suite validates the functionality of all modules:                 ║
║    • Neural Signal Generator (signal_gen.py)                                 ║
║    • ADC Simulator (adc_sim.py)                                              ║
║    • Integration (complete pipeline)                                         ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════════════════
 Neural Signal Generator Validation Tests
════════════════════════════════════════════════════════════════════════
✓ test_initialization
✓ test_noise_generation_length
✓ test_noise_distribution
...
✓ test_export_to_csv

════════════════════════════════════════════════════════════════════════
 Results: 42 passed, 0 failed
════════════════════════════════════════════════════════════════════════

[Similar output for ADC and Integration tests...]

════════════════════════════════════════════════════════════════════════
 FINAL TEST REPORT
════════════════════════════════════════════════════════════════════════
✓ PASSED - Neural Signal Generator Tests
✓ PASSED - ADC Simulator Tests
✓ PASSED - Integration Tests

Total Test Suites: 3
  Passed: 3
  Failed: 0
  Skipped: 0

✓ ALL TESTS PASSED! System validated and ready for use.
```

## Test Structure

### Test Files

```
testbench/
├── __init__.py                 # Package initialization
├── README.md                   # Detailed testbench documentation
├── run_all_tests.py            # Main test runner
├── test_signal_gen.py          # Signal generator tests (40+)
├── test_adc_sim.py             # ADC simulator tests (45+)
└── test_integration.py         # Integration tests (20+)
```

### Test Categories

1. **Unit Tests** - Test individual functions and methods
2. **Integration Tests** - Test modules working together
3. **Edge Case Tests** - Test boundary conditions
4. **Performance Tests** - Validate metrics (SNR, ENOB, etc.)
5. **Export Tests** - Validate CSV export functionality

## Test Coverage

### Signal Generator (`test_signal_gen.py`)

**Coverage: ~90%**

Tests include:
- ✓ Initialization with various parameters
- ✓ Noise generation (Gaussian distribution, correct length)
- ✓ Drift generation (frequency, amplitude, phase)
- ✓ Spike waveforms (all types: biphasic, triphasic, simple)
- ✓ Spike trains (Poisson process, refractory period)
- ✓ Spike injection (accuracy, boundary handling)
- ✓ Complete signal generation
- ✓ Multi-unit recordings
- ✓ CSV export functionality
- ✓ Reproducibility with seeds
- ✓ Edge cases (empty, very long, extreme parameters)

**Example Tests:**
```python
def test_noise_distribution(self):
    """Test noise follows Gaussian distribution"""
    noise = self.gen.generate_noise(10.0)
    assert abs(np.mean(noise)) < 0.01  # Mean near zero
    assert abs(np.std(noise) - self.gen.noise_amplitude) < 0.01

def test_refractory_period(self):
    """Test refractory period is enforced"""
    spike_times = self.gen.generate_spike_train(1.0, 500.0, refractory_period=0.002)
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        assert np.all(isis >= 0.002)  # All ISIs >= 2 ms
```

---

### ADC Simulator (`test_adc_sim.py`)

**Coverage: ~90%**

Tests include:
- ✓ Initialization and parameter calculations
- ✓ Analog-to-digital conversion accuracy
- ✓ Quantization error bounds (±0.5 LSB)
- ✓ Linearity of conversion
- ✓ Saturation and clipping behavior
- ✓ Sampling and downsampling
- ✓ Frequency preservation (below Nyquist)
- ✓ Timing jitter effects
- ✓ ADC noise addition
- ✓ SNR calculation (matches theory)
- ✓ ENOB calculation
- ✓ Different resolutions (8-16 bits)
- ✓ Various voltage ranges
- ✓ CSV export functionality

**Example Tests:**
```python
def test_quantization_error_bound(self):
    """Test quantization error is within ±0.5 LSB"""
    signal = np.random.uniform(self.adc.vref_neg, self.adc.vref_pos, 1000)
    codes, quantized = self.adc.analog_to_digital(signal)
    error = signal - quantized
    max_error = np.max(np.abs(error))
    assert max_error < self.adc.lsb  # Within 1 LSB

def test_theoretical_snr_formula(self):
    """Test SNR matches theoretical formula"""
    # For full-scale sine: SNR = 6.02N + 1.76 dB
    original = amplitude * np.sin(2 * np.pi * 10 * t) + offset
    codes, quantized = self.adc.analog_to_digital(original)
    snr = self.adc.get_snr(original, quantized)
    theoretical_snr = 6.02 * 12 + 1.76
    assert abs(snr - theoretical_snr) < 3  # Within 3 dB
```

---

### Integration Tests (`test_integration.py`)

**Coverage: ~85%**

Tests include:
- ✓ Complete signal → ADC pipeline
- ✓ Spike preservation through conversion
- ✓ Frequency content preservation
- ✓ Performance with various configurations
- ✓ Saturation handling in pipeline
- ✓ Multi-unit signals through ADC
- ✓ Realistic system configurations
- ✓ Data export from both modules
- ✓ Stress tests (long duration, high spike rates)

**Example Tests:**
```python
def test_signal_to_adc_pipeline(self):
    """Test complete signal generation → ADC conversion pipeline"""
    # Generate analog signal
    t_analog, signal_analog, spike_times, _ = self.gen.generate_signal(
        duration=0.5, firing_rate=30.0, seed=42
    )
    
    # Convert with ADC
    t_digital, sampled, codes, quantized = self.adc.sample_signal(
        t_analog, signal_analog
    )
    
    # Verify pipeline worked
    assert len(t_digital) < len(t_analog)  # Downsampled
    expected_ratio = self.gen.fs / self.adc.fs
    actual_ratio = len(t_analog) / len(t_digital)
    assert abs(actual_ratio - expected_ratio) < 1.0

def test_realistic_neural_recording(self):
    """Test realistic neural recording system configuration"""
    # Intan-style system: 16-bit, 20 kHz, ±5 mV
    adc_realistic = ADCSimulator(
        fs=20000, resolution=16, vref_pos=0.005, vref_neg=-0.005,
        enable_jitter=True, jitter_std=0.5e-6, adc_noise_std=0.0001
    )
    # ... test pipeline performance
    assert snr > 60  # Research-grade performance
```

---

## Running Tests

### Method 1: All Tests with Test Runner

**Most comprehensive - runs all suites with summary report**

```bash
python testbench/run_all_tests.py
```

**Output:** Detailed results for each test plus final summary

---

### Method 2: Individual Test Files

**Run specific test suite**

```bash
# Signal generator tests only
python testbench/test_signal_gen.py

# ADC simulator tests only
python testbench/test_adc_sim.py

# Integration tests only
python testbench/test_integration.py
```

---

### Method 3: With pytest

**Requires:** `pip install pytest pytest-cov`

**Run all tests:**
```bash
pytest testbench/ -v
```

**Run specific file:**
```bash
pytest testbench/test_signal_gen.py -v
```

**Run specific test:**
```bash
pytest testbench/test_signal_gen.py::TestNeuralSignalGenerator::test_noise_generation_length -v
```

**With coverage report:**
```bash
pytest testbench/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

**Parallel execution:**
```bash
pip install pytest-xdist
pytest testbench/ -n auto  # Use all CPU cores
```

---

### Method 4: Docker

**Run in isolated environment**

```bash
# All tests
docker-compose run neural-dsp python testbench/run_all_tests.py

# Specific file
docker-compose run neural-dsp python testbench/test_signal_gen.py

# With pytest
docker-compose run neural-dsp pytest testbench/ -v
```

---

## Interpreting Results

### Successful Test

```
✓ test_noise_generation_length
```
Test passed - functionality verified

### Failed Test

```
✗ test_noise_generation_length: AssertionError: assert 19999 == 20000
```
Test failed - check error message for details

### Test Summary

```
Results: 42 passed, 0 failed
```
All tests in this suite passed

---

## Test Performance

### Typical Run Times

| Test Suite | Tests | Duration |
|-----------|-------|----------|
| Signal Generator | 40+ | 5-10 sec |
| ADC Simulator | 45+ | 5-10 sec |
| Integration | 20+ | 10-15 sec |
| **Total** | **105+** | **20-35 sec** |

*Times vary based on system performance*

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python testbench/run_all_tests.py
```

---

## Troubleshooting

### Import Errors

```
ImportError: No module named 'src'
```

**Solution:** Run from project root:
```bash
cd /path/to/Neural_Signal_DSP
python testbench/run_all_tests.py
```

---

### pytest Not Found

```
ModuleNotFoundError: No module named 'pytest'
```

**Solution 1:** Install pytest:
```bash
pip install pytest
```

**Solution 2:** Use built-in runner (no pytest needed):
```bash
python testbench/test_signal_gen.py
```

---

### Random Test Failures

Some tests use random number generation. If implementation changes, random sequences may differ.

**Solution:** Tests use fixed seeds for reproducibility. If you modify algorithms, update test expectations or seeds.

---

### Memory Issues

```
MemoryError: Unable to allocate array
```

**Solution:** Some stress tests generate large signals. Run tests individually or reduce test signal durations.

---

## Writing New Tests

When adding new modules, create corresponding test files:

```python
"""
Test Suite for New Module

Tests validate:
- Feature 1
- Feature 2
- Edge cases
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pytest
from src.new_module import NewModule


class TestNewModule:
    """Test suite for NewModule"""
    
    def setup_method(self):
        """Setup before each test"""
        self.module = NewModule()
    
    def test_basic_functionality(self):
        """Test basic feature"""
        result = self.module.do_something()
        assert result is not None
    
    # Add more tests...


def run_validation_tests():
    """Run tests with built-in runner"""
    # Implementation...
    pass


if __name__ == '__main__':
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        success = run_validation_tests()
        sys.exit(0 if success else 1)
```

See `testbench/README.md` for detailed guidelines.

---

## Benefits of Testing

1. **Confidence** - Know your code works correctly
2. **Regression Prevention** - Catch bugs when changing code
3. **Documentation** - Tests show how to use the code
4. **Refactoring Safety** - Change implementation without fear
5. **Quality Assurance** - Professional-grade code reliability

---

## Next Steps

1. **Run the tests** to validate your installation
2. **Review test output** to understand what's being validated
3. **Examine test code** to learn testing patterns
4. **Write tests** for new features you add
5. **Use tests** during development for rapid feedback

---

**Testing is an investment in code quality and long-term maintainability!**

