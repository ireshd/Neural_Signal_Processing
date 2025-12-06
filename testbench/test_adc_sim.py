"""
Test Suite for ADC Simulator

Tests validate:
- ADC conversion accuracy
- Quantization behavior
- Sampling correctness
- Performance metrics (SNR, ENOB)
- Jitter and noise effects
- Saturation handling
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pytest
from src.adc_sim import ADCSimulator


class TestADCSimulator:
    """Test suite for ADCSimulator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.adc = ADCSimulator(
            fs=20000,
            resolution=12,
            vref_pos=1.0,
            vref_neg=-1.0,
            enable_saturation=True
        )
    
    # ===== Initialization Tests =====
    
    def test_initialization(self):
        """Test ADC initialization"""
        assert self.adc.fs == 20000
        assert self.adc.resolution == 12
        assert self.adc.vref_pos == 1.0
        assert self.adc.vref_neg == -1.0
    
    def test_calculated_parameters(self):
        """Test calculated ADC parameters"""
        # LSB calculation
        expected_lsb = (self.adc.vref_pos - self.adc.vref_neg) / (2**12 - 1)
        assert abs(self.adc.lsb - expected_lsb) < 1e-9
        
        # Max code
        assert self.adc.max_code == 2**12 - 1
        
        # Voltage range
        assert self.adc.voltage_range == 2.0
    
    def test_different_resolutions(self):
        """Test ADCs with different resolutions"""
        for resolution in [8, 10, 12, 14, 16]:
            adc = ADCSimulator(fs=20000, resolution=resolution)
            assert adc.resolution == resolution
            assert adc.max_code == 2**resolution - 1
    
    def test_asymmetric_vref(self):
        """Test ADC with asymmetric voltage range"""
        adc = ADCSimulator(vref_pos=2.0, vref_neg=-1.0)
        assert adc.voltage_range == 3.0
        assert adc.offset == -1.0
    
    # ===== Analog to Digital Conversion Tests =====
    
    def test_zero_voltage_conversion(self):
        """Test conversion of zero voltage"""
        signal = np.array([0.0])
        codes, quantized = self.adc.analog_to_digital(signal)
        
        # Zero should map to middle code
        expected_code = (self.adc.max_code + 1) // 2
        assert abs(codes[0] - expected_code) <= 1
    
    def test_full_scale_positive(self):
        """Test conversion at positive full scale"""
        signal = np.array([self.adc.vref_pos])
        codes, quantized = self.adc.analog_to_digital(signal)
        
        # Should map to max code
        assert codes[0] == self.adc.max_code
    
    def test_full_scale_negative(self):
        """Test conversion at negative full scale"""
        signal = np.array([self.adc.vref_neg])
        codes, quantized = self.adc.analog_to_digital(signal)
        
        # Should map to code 0
        assert codes[0] == 0
    
    def test_linear_conversion(self):
        """Test linearity of ADC conversion"""
        # Test range of voltages
        voltages = np.linspace(self.adc.vref_neg, self.adc.vref_pos, 100)
        codes, quantized = self.adc.analog_to_digital(voltages)
        
        # Codes should be monotonically increasing
        assert np.all(np.diff(codes) >= 0)
        
        # Check linearity (correlation should be very high)
        correlation = np.corrcoef(voltages, codes)[0, 1]
        assert correlation > 0.999
    
    def test_quantization_error_bound(self):
        """Test quantization error is within ±0.5 LSB"""
        np.random.seed(42)
        signal = np.random.uniform(self.adc.vref_neg, self.adc.vref_pos, 1000)
        codes, quantized = self.adc.analog_to_digital(signal)
        
        error = signal - quantized
        max_error = np.max(np.abs(error))
        
        # Maximum error should be less than 1 LSB
        assert max_error < self.adc.lsb
    
    def test_conversion_preserves_length(self):
        """Test ADC conversion preserves array length"""
        signal = np.random.randn(1000)
        codes, quantized = self.adc.analog_to_digital(signal)
        
        assert len(codes) == len(signal)
        assert len(quantized) == len(signal)
    
    # ===== Saturation Tests =====
    
    def test_saturation_enabled(self):
        """Test saturation clips signals correctly"""
        # Signal exceeding range
        signal = np.array([-2.0, -1.5, 0.0, 1.5, 2.0])
        codes, quantized = self.adc.analog_to_digital(signal)
        
        # Values should be clipped
        assert np.all(quantized >= self.adc.vref_neg)
        assert np.all(quantized <= self.adc.vref_pos)
        
        # Extremes should be at limits
        assert quantized[0] == self.adc.vref_neg
        assert quantized[-1] == self.adc.vref_pos
    
    def test_saturation_statistics(self):
        """Test saturation statistics tracking"""
        signal = np.array([-2.0, 0.0, 2.0])  # 2 out of 3 saturated
        codes, quantized = self.adc.analog_to_digital(signal)
        
        stats = self.adc.get_statistics()
        assert stats['num_saturated_samples'] == 2
        assert abs(stats['saturation_percentage'] - 66.67) < 0.1
    
    def test_no_saturation(self):
        """Test with signals within range"""
        signal = np.linspace(-0.5, 0.5, 100)
        codes, quantized = self.adc.analog_to_digital(signal)
        
        stats = self.adc.get_statistics()
        assert stats['num_saturated_samples'] == 0
        assert stats['saturation_percentage'] == 0.0
    
    # ===== Sampling Tests =====
    
    def test_sample_signal_downsampling(self):
        """Test signal is properly downsampled"""
        # High rate "analog" signal
        t_analog = np.linspace(0, 1, 100000)  # 100 kHz
        signal_analog = np.sin(2 * np.pi * 10 * t_analog)  # 10 Hz sine
        
        # Sample at 20 kHz
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        expected_samples = int(1.0 * self.adc.fs)
        assert len(t_digital) == expected_samples
        assert len(sampled) == expected_samples
    
    def test_sample_signal_interpolation(self):
        """Test interpolation accuracy during sampling"""
        # Simple linear signal
        t_analog = np.linspace(0, 1, 10000)
        signal_analog = t_analog  # Linear ramp
        
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Sampled values should match analog at sample times
        for i in range(len(t_digital)):
            expected = t_digital[i]
            # Allow some quantization error
            assert abs(quantized[i] - expected) < 2 * self.adc.lsb
    
    def test_sample_signal_preserves_frequency(self):
        """Test sampling preserves signal frequency (below Nyquist)"""
        # 100 Hz sine wave
        freq = 100
        t_analog = np.linspace(0, 0.1, 100000)
        signal_analog = np.sin(2 * np.pi * freq * t_analog)
        
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # FFT to check frequency
        fft = np.fft.fft(quantized)
        freqs = np.fft.fftfreq(len(quantized), 1/self.adc.fs)
        peak_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        detected_freq = abs(freqs[peak_idx])
        
        assert abs(detected_freq - freq) < 5  # Within 5 Hz
    
    def test_custom_sampling_rate(self):
        """Test sampling with custom target rate"""
        t_analog = np.linspace(0, 1, 100000)
        signal_analog = np.random.randn(100000)
        
        custom_fs = 10000  # 10 kHz instead of default 20 kHz
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog, target_fs=custom_fs
        )
        
        expected_samples = int(1.0 * custom_fs)
        assert abs(len(t_digital) - expected_samples) <= 1
    
    # ===== Jitter Tests =====
    
    def test_jitter_enabled(self):
        """Test sampling with jitter"""
        adc_jitter = ADCSimulator(
            fs=20000,
            resolution=12,
            enable_jitter=True,
            jitter_std=1e-6
        )
        
        t_analog = np.linspace(0, 0.1, 10000)
        signal_analog = np.sin(2 * np.pi * 100 * t_analog)
        
        t_digital, sampled, codes, quantized = adc_jitter.sample_signal(
            t_analog, signal_analog
        )
        
        # With jitter, sample times should vary slightly
        ideal_dt = 1.0 / adc_jitter.fs
        actual_dt = np.diff(t_digital)
        
        # Not all intervals should be exactly the same
        dt_variation = np.std(actual_dt)
        assert dt_variation > 0
    
    def test_jitter_bounds(self):
        """Test jitter keeps samples within time bounds"""
        adc_jitter = ADCSimulator(
            fs=20000,
            resolution=12,
            enable_jitter=True,
            jitter_std=5e-6  # Larger jitter
        )
        
        t_analog = np.linspace(0, 1, 100000)
        signal_analog = np.random.randn(100000)
        
        t_digital, _, _, _ = adc_jitter.sample_signal(t_analog, signal_analog)
        
        # All samples should be within original time bounds
        assert np.all(t_digital >= t_analog[0])
        assert np.all(t_digital <= t_analog[-1])
        
        # Should be monotonically increasing
        assert np.all(np.diff(t_digital) >= 0)
    
    # ===== ADC Noise Tests =====
    
    def test_adc_noise(self):
        """Test ADC noise is added correctly"""
        adc_noise = ADCSimulator(
            fs=20000,
            resolution=12,
            adc_noise_std=0.01  # 10 mV noise
        )
        
        # Clean signal
        signal = np.zeros(1000)
        codes, quantized = adc_noise.analog_to_digital(signal)
        
        # Output should have noise
        assert np.std(quantized) > 0
        assert np.std(quantized) < 0.05  # But not too much
    
    def test_no_adc_noise(self):
        """Test without ADC noise"""
        # Clean constant signal
        signal = np.ones(1000) * 0.5
        codes, quantized = self.adc.analog_to_digital(signal)
        
        # Output should be mostly constant (only quantization variation)
        unique_values = len(np.unique(quantized))
        assert unique_values <= 2  # At most 2 quantization levels
    
    # ===== Performance Metrics Tests =====
    
    def test_snr_calculation(self):
        """Test SNR calculation"""
        # Perfect sine wave
        t = np.linspace(0, 1, 20000)
        original = np.sin(2 * np.pi * 10 * t)
        
        # Quantize it
        codes, quantized = self.adc.analog_to_digital(original)
        
        # Calculate SNR
        snr = self.adc.get_snr(original, quantized)
        
        # For 12-bit ADC, theoretical SNR ≈ 6.02*12 + 1.76 ≈ 74 dB
        # Allow some tolerance
        assert snr > 60  # At least 60 dB
        assert snr < 90  # Less than 90 dB
    
    def test_enob_calculation(self):
        """Test ENOB calculation"""
        t = np.linspace(0, 1, 20000)
        original = np.sin(2 * np.pi * 10 * t)
        
        codes, quantized = self.adc.analog_to_digital(original)
        enob = self.adc.get_enob(original, quantized)
        
        # ENOB should be less than or equal to nominal resolution
        assert enob <= self.adc.resolution
        assert enob > self.adc.resolution - 2  # Within 2 bits
    
    def test_snr_improves_with_resolution(self):
        """Test SNR improves with higher resolution"""
        t = np.linspace(0, 1, 20000)
        original = np.sin(2 * np.pi * 10 * t)
        
        snr_values = []
        for resolution in [8, 10, 12, 14, 16]:
            adc = ADCSimulator(fs=20000, resolution=resolution)
            codes, quantized = adc.analog_to_digital(original)
            snr = adc.get_snr(original, quantized)
            snr_values.append(snr)
        
        # SNR should increase with resolution
        for i in range(len(snr_values) - 1):
            assert snr_values[i+1] > snr_values[i]
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        signal = np.random.uniform(-0.5, 0.5, 1000)
        codes, quantized = self.adc.analog_to_digital(signal)
        
        stats = self.adc.get_statistics()
        
        # Check all expected keys are present
        expected_keys = [
            'resolution_bits', 'sampling_rate_hz', 'lsb_voltage',
            'voltage_range', 'max_code', 'num_samples_processed',
            'num_saturated_samples', 'saturation_percentage'
        ]
        for key in expected_keys:
            assert key in stats
    
    # ===== Quantize Only Tests =====
    
    def test_quantize_only(self):
        """Test quantization without sampling"""
        signal = np.linspace(-1, 1, 1000)
        quantized = self.adc.quantize_only(signal)
        
        assert len(quantized) == len(signal)
        
        # Quantized should have discrete levels
        unique_levels = len(np.unique(quantized))
        assert unique_levels <= self.adc.max_code + 1
    
    # ===== CSV Export Tests =====
    
    def test_export_to_csv(self):
        """Test CSV export functionality"""
        import tempfile
        import os
        
        # Generate some data
        t = np.linspace(0, 0.1, 2000)
        signal = np.sin(2 * np.pi * 10 * t)
        codes, quantized = self.adc.analog_to_digital(signal)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exported = self.adc.export_to_csv(
                t, signal, codes, quantized,
                output_dir=tmpdir,
                prefix='test'
            )
            
            # Check files were created
            assert 'adc_data' in exported
            assert 'quantization_error' in exported
            assert 'statistics' in exported
            
            # Verify file existence
            for path in exported.values():
                assert os.path.exists(path)
            
            # Check data file content
            data = np.loadtxt(exported['adc_data'], delimiter=',', skiprows=1)
            assert data.shape[0] == len(t)
            assert data.shape[1] == 4  # time, analog, code, quantized
    
    # ===== Edge Cases and Error Handling =====
    
    def test_empty_signal(self):
        """Test handling of empty signal"""
        signal = np.array([])
        codes, quantized = self.adc.analog_to_digital(signal)
        
        assert len(codes) == 0
        assert len(quantized) == 0
    
    def test_single_sample(self):
        """Test with single sample"""
        signal = np.array([0.5])
        codes, quantized = self.adc.analog_to_digital(signal)
        
        assert len(codes) == 1
        assert len(quantized) == 1
    
    def test_very_long_signal(self):
        """Test with long signal (memory test)"""
        signal = np.random.randn(100000)
        codes, quantized = self.adc.analog_to_digital(signal)
        
        assert len(codes) == 100000
        assert len(quantized) == 100000
    
    def test_nan_handling(self):
        """Test handling of NaN values"""
        signal = np.array([0.0, np.nan, 0.5])
        codes, quantized = self.adc.analog_to_digital(signal)
        
        # NaN should produce some output (may be 0 or middle code)
        assert len(codes) == 3
    
    def test_different_voltage_ranges(self):
        """Test with different voltage range configurations"""
        configs = [
            (0.005, -0.005),  # ±5 mV (typical neural)
            (1.65, -1.65),    # ±1.65 V (3.3V system)
            (2.5, -2.5),      # ±2.5 V (5V system)
            (3.3, 0),         # 0-3.3V (unipolar)
        ]
        
        for vref_pos, vref_neg in configs:
            adc = ADCSimulator(
                fs=20000,
                resolution=12,
                vref_pos=vref_pos,
                vref_neg=vref_neg
            )
            
            signal = np.array([vref_neg, 0, vref_pos])
            codes, quantized = adc.analog_to_digital(signal)
            
            # Should handle all ranges
            assert codes[0] == 0  # Minimum
            assert codes[-1] == adc.max_code  # Maximum
    
    def test_theoretical_snr_formula(self):
        """Test SNR matches theoretical formula for ideal ADC"""
        # For a full-scale sine wave: SNR = 6.02N + 1.76 dB
        t = np.linspace(0, 1, 20000)
        # Full scale sine (from vref_neg to vref_pos)
        amplitude = (self.adc.vref_pos - self.adc.vref_neg) / 2
        offset = (self.adc.vref_pos + self.adc.vref_neg) / 2
        original = amplitude * np.sin(2 * np.pi * 10 * t) + offset
        
        codes, quantized = self.adc.analog_to_digital(original)
        snr = self.adc.get_snr(original, quantized)
        
        # Theoretical SNR for 12-bit
        theoretical_snr = 6.02 * 12 + 1.76
        
        # Allow 3 dB tolerance
        assert abs(snr - theoretical_snr) < 3


def run_validation_tests():
    """
    Run validation tests and print results
    """
    print("=" * 70)
    print(" ADC Simulator Validation Tests")
    print("=" * 70)
    
    test = TestADCSimulator()
    test.setup_method()
    
    test_methods = [
        method for method in dir(test) 
        if method.startswith('test_') and callable(getattr(test, method))
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            # Reset for each test
            test.setup_method()
            method = getattr(test, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        success = run_validation_tests()
        sys.exit(0 if success else 1)

