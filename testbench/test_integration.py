"""
Integration Tests

Tests validate:
- Signal generator → ADC pipeline
- Data flow between modules
- End-to-end functionality
- Performance of combined system
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pytest
from src.signal_gen import NeuralSignalGenerator
from src.adc_sim import ADCSimulator


class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gen = NeuralSignalGenerator(
            fs=100000,  # High rate for "analog"
            noise_amplitude=0.02,
            drift_amplitude=0.1
        )
        
        self.adc = ADCSimulator(
            fs=20000,  # Standard sampling rate
            resolution=12,
            vref_pos=1.0,
            vref_neg=-1.0
        )
    
    # ===== Basic Pipeline Tests =====
    
    def test_signal_to_adc_pipeline(self):
        """Test complete signal generation → ADC conversion pipeline"""
        # Generate analog signal
        duration = 0.5
        t_analog, signal_analog, spike_times, spike_indices = self.gen.generate_signal(
            duration=duration,
            firing_rate=30.0,
            seed=42
        )
        
        # Convert with ADC
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Verify pipeline worked
        assert len(t_digital) > 0
        assert len(codes) > 0
        assert len(quantized) > 0
        
        # Digital samples should be fewer than analog
        assert len(t_digital) < len(t_analog)
        
        # Downsampling ratio should match fs ratio
        expected_ratio = self.gen.fs / self.adc.fs
        actual_ratio = len(t_analog) / len(t_digital)
        assert abs(actual_ratio - expected_ratio) < 1.0
    
    def test_spike_preservation(self):
        """Test that spikes are preserved through ADC conversion"""
        # Generate signal with known spikes
        duration = 1.0
        t_analog, signal_analog, spike_times, spike_indices = self.gen.generate_signal(
            duration=duration,
            firing_rate=20.0,
            seed=42
        )
        
        # Convert
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Check that digitized signal has high-amplitude events
        # corresponding to spikes
        threshold = np.std(quantized) * 2
        digital_peaks = np.where(np.abs(quantized) > threshold)[0]
        
        # Should detect some peaks
        assert len(digital_peaks) > 0
        
        # Number of detected peaks should be roughly similar to spike count
        # (allowing for some loss due to sampling and quantization)
        assert len(digital_peaks) >= len(spike_times) * 0.5
    
    def test_frequency_content_preservation(self):
        """Test that frequency content is preserved (below Nyquist)"""
        # Generate signal with known frequency component
        duration = 1.0
        t_analog, signal_analog, _, _ = self.gen.generate_signal(
            duration=duration,
            firing_rate=10.0,
            seed=42
        )
        
        # ADC conversion
        t_digital, _, _, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Check drift frequency is preserved (1 Hz)
        fft_digital = np.fft.fft(quantized)
        freqs = np.fft.fftfreq(len(quantized), 1/self.adc.fs)
        
        # Find peaks in 0.5-2 Hz range (around 1 Hz drift)
        freq_mask = (np.abs(freqs) > 0.5) & (np.abs(freqs) < 2.0)
        peak_in_range = np.any(np.abs(fft_digital[freq_mask]) > np.median(np.abs(fft_digital)) * 2)
        
        assert peak_in_range
    
    # ===== Performance Tests =====
    
    def test_adc_snr_with_neural_signal(self):
        """Test ADC achieves reasonable SNR with neural signals"""
        duration = 0.5
        t_analog, signal_analog, _, _ = self.gen.generate_signal(
            duration=duration,
            firing_rate=25.0,
            seed=42
        )
        
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        snr = self.adc.get_snr(sampled, quantized)
        enob = self.adc.get_enob(sampled, quantized)
        
        # Should achieve reasonable performance
        assert snr > 50  # At least 50 dB SNR
        assert enob > 10  # At least 10 effective bits
    
    def test_different_resolutions_performance(self):
        """Test pipeline with different ADC resolutions"""
        duration = 0.2
        t_analog, signal_analog, _, _ = self.gen.generate_signal(
            duration=duration,
            firing_rate=30.0,
            seed=42
        )
        
        snr_values = []
        for resolution in [8, 12, 16]:
            adc = ADCSimulator(
                fs=20000,
                resolution=resolution,
                vref_pos=1.0,
                vref_neg=-1.0
            )
            
            t_digital, sampled, codes, quantized = adc.sample_signal(
                t_analog, signal_analog
            )
            
            snr = adc.get_snr(sampled, quantized)
            snr_values.append(snr)
        
        # SNR should improve with resolution
        assert snr_values[1] > snr_values[0]  # 12-bit > 8-bit
        assert snr_values[2] > snr_values[1]  # 16-bit > 12-bit
    
    # ===== Saturation Tests =====
    
    def test_large_signal_saturation(self):
        """Test ADC handles large signals with saturation"""
        # Generate large signal
        gen_large = NeuralSignalGenerator(
            fs=100000,
            noise_amplitude=0.1,
            drift_amplitude=0.5  # Large drift
        )
        
        duration = 0.2
        t_analog, signal_analog, _, _ = gen_large.generate_signal(
            duration=duration,
            firing_rate=50.0,
            seed=42
        )
        
        # Scale to exceed ADC range
        signal_analog = signal_analog * 2.0
        
        # Convert
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Check saturation occurred
        stats = self.adc.get_statistics()
        assert stats['saturation_percentage'] > 0
        
        # All quantized values should be within range
        assert np.all(quantized >= self.adc.vref_neg)
        assert np.all(quantized <= self.adc.vref_pos)
    
    def test_optimal_signal_range(self):
        """Test with signal optimally scaled for ADC range"""
        # Generate signal
        duration = 0.5
        t_analog, signal_analog, _, _ = self.gen.generate_signal(
            duration=duration,
            firing_rate=30.0,
            seed=42
        )
        
        # Scale to use 80% of ADC range
        signal_peak = np.max(np.abs(signal_analog))
        target_peak = self.adc.vref_pos * 0.8
        scale_factor = target_peak / signal_peak
        signal_scaled = signal_analog * scale_factor
        
        # Convert
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_scaled
        )
        
        # Should have minimal saturation
        stats = self.adc.get_statistics()
        assert stats['saturation_percentage'] < 1.0
        
        # Should achieve good SNR
        snr = self.adc.get_snr(sampled, quantized)
        assert snr > 60
    
    # ===== Multi-Unit Tests =====
    
    def test_multi_unit_through_adc(self):
        """Test multi-unit signal through ADC"""
        duration = 0.5
        num_units = 3
        
        t_analog, signal_analog, all_spike_times = self.gen.generate_multi_unit_signal(
            duration=duration,
            num_units=num_units,
            firing_rates=[20.0, 30.0, 40.0],
            seed=42
        )
        
        # Convert
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Should successfully convert
        assert len(quantized) > 0
        
        # Total spike count
        total_spikes = sum(len(spikes) for spikes in all_spike_times)
        
        # Digital signal should have high-activity regions
        activity = np.abs(quantized - np.mean(quantized))
        high_activity_samples = np.sum(activity > np.std(activity))
        
        # Should have reasonable number of high-activity samples
        assert high_activity_samples > total_spikes * 0.3
    
    # ===== Different Configurations Tests =====
    
    def test_low_sampling_rate(self):
        """Test with lower ADC sampling rate"""
        adc_low = ADCSimulator(fs=10000, resolution=12)  # 10 kHz
        
        duration = 0.2
        t_analog, signal_analog, _, _ = self.gen.generate_signal(
            duration=duration,
            firing_rate=20.0,
            seed=42
        )
        
        t_digital, sampled, codes, quantized = adc_low.sample_signal(
            t_analog, signal_analog
        )
        
        # Should still work, just fewer samples
        expected_samples = int(duration * 10000)
        assert abs(len(t_digital) - expected_samples) <= 1
    
    def test_high_resolution_adc(self):
        """Test with high-resolution ADC"""
        adc_high = ADCSimulator(
            fs=20000,
            resolution=16,  # 16-bit
            vref_pos=1.0,
            vref_neg=-1.0
        )
        
        duration = 0.2
        t_analog, signal_analog, _, _ = self.gen.generate_signal(
            duration=duration,
            firing_rate=25.0,
            seed=42
        )
        
        t_digital, sampled, codes, quantized = adc_high.sample_signal(
            t_analog, signal_analog
        )
        
        # Should achieve better performance
        snr = adc_high.get_snr(sampled, quantized)
        enob = adc_high.get_enob(sampled, quantized)
        
        assert snr > 70  # Higher SNR
        assert enob > 14  # Higher ENOB
    
    # ===== Realistic System Tests =====
    
    def test_realistic_neural_recording(self):
        """Test realistic neural recording system configuration"""
        # Intan-style system
        gen_realistic = NeuralSignalGenerator(
            fs=100000,
            noise_amplitude=0.01,  # Low noise
            drift_amplitude=0.02
        )
        
        adc_realistic = ADCSimulator(
            fs=20000,
            resolution=16,
            vref_pos=0.005,  # ±5 mV
            vref_neg=-0.005,
            enable_jitter=True,
            jitter_std=0.5e-6,
            adc_noise_std=0.0001
        )
        
        duration = 1.0
        t_analog, signal_analog, spike_times, _ = gen_realistic.generate_signal(
            duration=duration,
            firing_rate=25.0,
            seed=42
        )
        
        # Scale to realistic amplitudes
        signal_analog = signal_analog * 0.005
        
        # Convert
        t_digital, sampled, codes, quantized = adc_realistic.sample_signal(
            t_analog, signal_analog
        )
        
        # Check system performance
        snr = adc_realistic.get_snr(sampled, quantized)
        stats = adc_realistic.get_statistics()
        
        # Should achieve research-grade performance
        assert snr > 60
        assert stats['saturation_percentage'] < 0.1
        assert len(t_digital) == 20000  # 1 second at 20 kHz
    
    def test_data_export_pipeline(self):
        """Test complete data export from both modules"""
        import tempfile
        import os
        
        duration = 0.2
        t_analog, signal_analog, spike_times, spike_indices = self.gen.generate_signal(
            duration=duration,
            firing_rate=30.0,
            seed=42
        )
        
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export from signal generator
            signal_files = self.gen.export_to_csv(
                t_analog, signal_analog, spike_times, spike_indices,
                output_dir=tmpdir,
                prefix='test_signal'
            )
            
            # Export from ADC
            adc_files = self.adc.export_to_csv(
                t_digital, sampled, codes, quantized,
                output_dir=tmpdir,
                prefix='test_adc'
            )
            
            # Verify all files exist
            for files in [signal_files, adc_files]:
                for path in files.values():
                    assert os.path.exists(path)
            
            # Load and verify consistency
            signal_data = np.loadtxt(
                signal_files['signal_data'],
                delimiter=',',
                skiprows=1
            )
            
            adc_data = np.loadtxt(
                adc_files['adc_data'],
                delimiter=',',
                skiprows=1
            )
            
            # Analog should have more samples
            assert len(signal_data) > len(adc_data)
    
    # ===== Stress Tests =====
    
    def test_long_duration_pipeline(self):
        """Test pipeline with longer duration"""
        duration = 5.0
        
        # Use lower fs for analog to save memory
        gen_test = NeuralSignalGenerator(fs=50000)
        
        t_analog, signal_analog, _, _ = gen_test.generate_signal(
            duration=duration,
            firing_rate=30.0,
            seed=42
        )
        
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Should complete successfully
        expected_digital_samples = int(duration * self.adc.fs)
        assert abs(len(t_digital) - expected_digital_samples) <= 1
    
    def test_high_spike_rate(self):
        """Test with very high spike rates"""
        duration = 0.5
        t_analog, signal_analog, spike_times, _ = self.gen.generate_signal(
            duration=duration,
            firing_rate=200.0,  # Very high rate
            seed=42
        )
        
        t_digital, sampled, codes, quantized = self.adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Should handle high spike rate
        assert len(quantized) > 0
        assert len(spike_times) > 50  # Should have many spikes


def run_integration_tests():
    """
    Run integration tests and print results
    """
    print("=" * 70)
    print(" Integration Tests")
    print("=" * 70)
    
    test = TestIntegration()
    test.setup_method()
    
    test_methods = [
        method for method in dir(test) 
        if method.startswith('test_') and callable(getattr(test, method))
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
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
        success = run_integration_tests()
        sys.exit(0 if success else 1)

