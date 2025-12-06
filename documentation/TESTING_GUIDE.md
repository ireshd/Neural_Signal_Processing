# Testing Guide

Complete guide to testing the Neural Signal DSP Pipeline.

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

## Test Structure

```
testbench/
├── test_signal_gen.py       # Signal generator tests (40+)
├── test_adc_sim.py           # ADC simulator tests (45+)
├── test_integration.py       # Integration tests (20+)
└── run_all_tests.py          # Test runner
```

**Total:** 105+ tests covering all implemented modules

## Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Signal Generator | 40+ | ~90% |
| ADC Simulator | 45+ | ~90% |
| Integration | 20+ | ~85% |

## Running Tests

### All Tests
```bash
python testbench/run_all_tests.py
```

### Individual Suites
```bash
python testbench/test_signal_gen.py
python testbench/test_adc_sim.py
python testbench/test_integration.py
```

### With pytest
```bash
pytest testbench/ -v --cov=src
```

## Expected Output

```
✓ ALL TESTS PASSED! System validated and ready for use.
```

## What's Tested

**Signal Generator:**
- Noise generation (distribution, length)
- Drift generation (frequency, amplitude)
- Spike waveforms (all types)
- Spike trains (Poisson, refractory period)
- CSV export
- Edge cases

**ADC Simulator:**
- Quantization accuracy (±0.5 LSB)
- Saturation handling
- Sampling and downsampling
- Jitter effects
- Performance metrics (SNR, ENOB)
- CSV export

**Integration:**
- Complete pipeline (signal → ADC)
- Spike preservation
- Frequency preservation
- Multi-unit through ADC
- Realistic configurations

## Continuous Integration

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: python testbench/run_all_tests.py
```

See `testbench/README.md` for detailed testing documentation.

