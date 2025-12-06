

# Test Bench Directory

Comprehensive test suite for validating all Neural Signal DSP modules.

## Overview

This directory contains unit tests, integration tests, and validation scripts to ensure code quality and correctness.

## Test Files

### 1. `test_signal_gen.py`
**Module:** Neural Signal Generator (`src/signal_gen.py`)

**Test Coverage:**
- ✓ Initialization and parameters
- ✓ Noise generation (distribution, length, amplitude)
- ✓ Drift generation (frequency, amplitude, phase)
- ✓ Spike waveform generation (all types)
- ✓ Spike train generation (Poisson process, refractory period)
- ✓ Spike injection
- ✓ Complete signal generation
- ✓ Multi-unit signals
- ✓ CSV export
- ✓ Edge cases and error handling

**Test Count:** 40+ tests

---

### 2. `test_adc_sim.py`
**Module:** ADC Simulator (`src/adc_sim.py`)

**Test Coverage:**
- ✓ Initialization and parameters
- ✓ Analog-to-digital conversion
- ✓ Quantization behavior
- ✓ Saturation and clipping
- ✓ Sampling and downsampling
- ✓ Timing jitter effects
- ✓ ADC noise
- ✓ Performance metrics (SNR, ENOB)
- ✓ CSV export
- ✓ Edge cases and different configurations

**Test Count:** 45+ tests

---

### 3. `test_integration.py`
**Module:** Complete Pipeline

**Test Coverage:**
- ✓ Signal → ADC pipeline
- ✓ Spike preservation through conversion
- ✓ Frequency content preservation
- ✓ Performance with different configurations
- ✓ Saturation handling
- ✓ Multi-unit through ADC
- ✓ Realistic system configurations
- ✓ Data export pipeline
- ✓ Stress tests

**Test Count:** 20+ tests

---

### 4. `run_all_tests.py`
**Test Runner Script**

Executes all test suites and generates comprehensive reports.

---

## Running Tests

### Option 1: Run All Tests (Recommended)

```bash
# Using Python
python testbench/run_all_tests.py
```

```bash
# Using Docker
docker-compose run neural-dsp python testbench/run_all_tests.py
```

This runs all test suites and provides a detailed report.

---

### Option 2: Run Individual Test Files

**Signal Generator Tests:**
```bash
python testbench/test_signal_gen.py
```

**ADC Simulator Tests:**
```bash
python testbench/test_adc_sim.py
```

**Integration Tests:**
```bash
python testbench/test_integration.py
```

---

### Option 3: Using pytest (If Available)

**Install pytest:**
```bash
pip install pytest
```

**Run all tests:**
```bash
pytest testbench/ -v
```

**Run specific test file:**
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
```

---

## Test Output

### Successful Test Run

```
═══════════════════════════════════════════════════════════════════════
 Neural Signal Generator Validation Tests
═══════════════════════════════════════════════════════════════════════
✓ test_initialization
✓ test_noise_generation_length
✓ test_noise_distribution
✓ test_drift_generation_length
...
✓ test_export_to_csv

═══════════════════════════════════════════════════════════════════════
 Results: 42 passed, 0 failed
═══════════════════════════════════════════════════════════════════════
```

### Failed Test

```
✗ test_noise_generation_length: AssertionError: assert 19999 == 20000
```

---

## Writing New Tests

### Test Structure

```python
class TestMyModule:
    """Test suite for MyModule"""
    
    def setup_method(self):
        """Setup before each test"""
        self.module = MyModule()
    
    def test_feature_name(self):
        """Test description"""
        # Arrange
        input_data = ...
        
        # Act
        result = self.module.do_something(input_data)
        
        # Assert
        assert result == expected_value
```

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `TestModuleName`
- Test methods: `test_<feature>_<scenario>`

### Good Test Practices

1. **Test one thing at a time**
   ```python
   def test_noise_generation_length(self):
       """Test ONLY that noise array has correct length"""
       noise = self.gen.generate_noise(1.0)
       assert len(noise) == 20000
   ```

2. **Use descriptive names**
   ```python
   # Good
   def test_spike_train_respects_refractory_period(self):
   
   # Bad
   def test_spikes(self):
   ```

3. **Test edge cases**
   ```python
   def test_empty_signal(self):
   def test_very_long_signal(self):
   def test_nan_handling(self):
   ```

4. **Use fixtures for setup**
   ```python
   def setup_method(self):
       """Runs before each test"""
       self.gen = NeuralSignalGenerator(fs=20000)
   ```

5. **Test both success and failure**
   ```python
   def test_valid_input(self):
       result = func(valid_data)
       assert result is not None
   
   def test_invalid_input(self):
       with pytest.raises(ValueError):
           func(invalid_data)
   ```

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
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest testbench/ -v --cov=src
```

---

## Test Coverage Goals

**Target Coverage:** >80% for each module

**Current Coverage:**
- `signal_gen.py`: ~90%
- `adc_sim.py`: ~90%
- Integration: ~85%

### Checking Coverage

```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
pytest testbench/ --cov=src --cov-report=term-missing

# Generate HTML report
pytest testbench/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Troubleshooting

### ImportError: No module named 'src'

Make sure you're running from the project root:
```bash
cd /path/to/Neural_Signal_DSP
python testbench/run_all_tests.py
```

### Tests fail with random seed issues

Some tests use random seeds for reproducibility. If you modify the implementation, random sequences may change. Update the expected values or seeds in tests.

### pytest not found

Install pytest:
```bash
pip install pytest
```

Or use the built-in test runner (no pytest required):
```bash
python testbench/test_signal_gen.py
```

### Memory issues with long tests

Some tests generate long signals. If you encounter memory issues:
- Run tests individually
- Reduce signal durations in stress tests
- Increase available memory

---

## Performance Benchmarks

### Typical Test Run Times

- `test_signal_gen.py`: ~5-10 seconds
- `test_adc_sim.py`: ~5-10 seconds
- `test_integration.py`: ~10-15 seconds
- **Total**: ~20-35 seconds

### Optimization Tips

- Use smaller test signals where possible
- Mock expensive operations if testing logic only
- Run slow tests in separate suite

---

## Adding Tests for New Modules

When you add a new module (e.g., `dma_buffer.py`), create `test_dma_buffer.py`:

```python
"""
Test Suite for DMA Buffer

Tests validate:
- Buffer initialization
- Circular buffer behavior
- Overflow handling
- ISR callbacks
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pytest
from src.dma_buffer import DMABuffer


class TestDMABuffer:
    """Test suite for DMABuffer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.buffer = DMABuffer(size=1024)
    
    def test_initialization(self):
        """Test buffer initialization"""
        assert self.buffer.size == 1024
        assert len(self.buffer.data) == 1024
    
    # Add more tests...


def run_validation_tests():
    """Run validation tests and print results"""
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

Then update `run_all_tests.py` to include the new test module.

---

## Best Practices Summary

1. ✓ **Write tests first** (TDD approach when possible)
2. ✓ **Test public APIs** thoroughly
3. ✓ **Include edge cases** (empty, very large, invalid inputs)
4. ✓ **Test error handling** (exceptions, invalid states)
5. ✓ **Keep tests independent** (don't rely on execution order)
6. ✓ **Use meaningful assertions** with clear messages
7. ✓ **Document what you're testing** in docstrings
8. ✓ **Run tests frequently** during development
9. ✓ **Maintain tests** when changing code
10. ✓ **Aim for >80% coverage** but focus on meaningful tests

---

## Resources

- **pytest documentation**: https://docs.pytest.org/
- **Python unittest**: https://docs.python.org/3/library/unittest.html
- **Test-Driven Development**: https://en.wikipedia.org/wiki/Test-driven_development

---

**Questions or issues with tests? Check the test output for detailed error messages.**

