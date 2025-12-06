"""
Test Suite for Neural Signal Generator

Tests validate:
- Signal generation correctness
- Spike injection accuracy
- Parameter handling
- Output formats
- Edge cases
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pytest
from src.signal_gen import NeuralSignalGenerator


class TestNeuralSignalGenerator:
    """Test suite for NeuralSignalGenerator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gen = NeuralSignalGenerator(
            fs=20000,
            noise_amplitude=0.05,
            drift_amplitude=0.2,
            drift_freq=1.0
        )
    
    # ===== Initialization Tests =====
    
    def test_initialization(self):
        """Test generator initialization"""
        assert self.gen.fs == 20000
        assert self.gen.noise_amplitude == 0.05
        assert self.gen.drift_amplitude == 0.2
        assert self.gen.drift_freq == 1.0
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        gen = NeuralSignalGenerator(fs=30000, noise_amplitude=0.1)
        assert gen.fs == 30000
        assert gen.noise_amplitude == 0.1
    
    # ===== Noise Generation Tests =====
    
    def test_noise_generation_length(self):
        """Test noise array has correct length"""
        duration = 1.0
        noise = self.gen.generate_noise(duration)
        expected_length = int(duration * self.gen.fs)
        assert len(noise) == expected_length
    
    def test_noise_distribution(self):
        """Test noise follows Gaussian distribution"""
        np.random.seed(42)
        noise = self.gen.generate_noise(10.0)  # Long signal for statistics
        
        # Check mean is close to zero
        assert abs(np.mean(noise)) < 0.01
        
        # Check std is close to noise_amplitude
        assert abs(np.std(noise) - self.gen.noise_amplitude) < 0.01
    
    def test_noise_different_durations(self):
        """Test noise generation with different durations"""
        for duration in [0.1, 0.5, 1.0, 2.0]:
            noise = self.gen.generate_noise(duration)
            expected_length = int(duration * self.gen.fs)
            assert len(noise) == expected_length
    
    # ===== Drift Generation Tests =====
    
    def test_drift_generation_length(self):
        """Test drift array has correct length"""
        duration = 1.0
        t, drift = self.gen.generate_drift(duration)
        expected_length = int(duration * self.gen.fs)
        assert len(t) == expected_length
        assert len(drift) == expected_length
    
    def test_drift_frequency(self):
        """Test drift has correct frequency"""
        duration = 2.0
        t, drift = self.gen.generate_drift(duration)
        
        # FFT to check frequency
        fft = np.fft.fft(drift)
        freqs = np.fft.fftfreq(len(drift), 1/self.gen.fs)
        
        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        peak_freq = abs(freqs[peak_idx])
        
        assert abs(peak_freq - self.gen.drift_freq) < 0.1
    
    def test_drift_amplitude(self):
        """Test drift has correct amplitude"""
        duration = 2.0
        t, drift = self.gen.generate_drift(duration)
        
        # Peak-to-peak amplitude should be 2 * drift_amplitude
        pp_amplitude = np.max(drift) - np.min(drift)
        expected_pp = 2 * self.gen.drift_amplitude
        
        assert abs(pp_amplitude - expected_pp) < 0.01
    
    def test_drift_phase(self):
        """Test drift with different phases"""
        duration = 1.0
        t1, drift1 = self.gen.generate_drift(duration, phase=0)
        t2, drift2 = self.gen.generate_drift(duration, phase=np.pi)
        
        # Signals with 180° phase difference should be opposite
        assert not np.allclose(drift1, drift2)
        assert np.allclose(drift1, -drift2, atol=0.01)
    
    # ===== Spike Waveform Tests =====
    
    def test_spike_waveform_length(self):
        """Test spike waveform has correct length"""
        spike = self.gen.generate_spike_waveform('biphasic')
        expected_length = int(0.002 * self.gen.fs)  # 2 ms
        assert len(spike) == expected_length
    
    def test_spike_types(self):
        """Test all spike types can be generated"""
        for spike_type in ['biphasic', 'triphasic', 'simple']:
            spike = self.gen.generate_spike_waveform(spike_type)
            assert len(spike) > 0
            assert not np.all(spike == 0)
    
    def test_biphasic_waveform_shape(self):
        """Test biphasic waveform has negative then positive phases"""
        spike = self.gen.generate_spike_waveform('biphasic')
        
        # Should have negative peak
        assert np.min(spike) < -0.5
        # Should have positive component
        assert np.max(spike) > 0.1
    
    def test_waveform_normalization(self):
        """Test spike waveforms are properly scaled"""
        spike = self.gen.generate_spike_waveform('biphasic')
        
        # Peak should be around -1.0 (negative peak)
        assert abs(np.min(spike) - (-1.0)) < 0.1
    
    # ===== Spike Train Generation Tests =====
    
    def test_spike_train_count(self):
        """Test spike train has approximately correct number of spikes"""
        np.random.seed(42)
        duration = 10.0
        firing_rate = 30.0
        
        spike_times = self.gen.generate_spike_train(duration, firing_rate)
        
        # Expected count with some tolerance (Poisson variability)
        expected_count = duration * firing_rate
        tolerance = 3 * np.sqrt(expected_count)  # 3 sigma
        
        assert abs(len(spike_times) - expected_count) < tolerance
    
    def test_spike_train_within_duration(self):
        """Test all spike times are within specified duration"""
        duration = 2.0
        spike_times = self.gen.generate_spike_train(duration, 25.0)
        
        assert np.all(spike_times >= 0)
        assert np.all(spike_times <= duration)
    
    def test_refractory_period(self):
        """Test refractory period is enforced"""
        duration = 1.0
        firing_rate = 500.0  # Very high rate to test refractory period
        refractory = 0.002  # 2 ms
        
        spike_times = self.gen.generate_spike_train(
            duration, firing_rate, refractory_period=refractory
        )
        
        # Check all inter-spike intervals
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            assert np.all(isis >= refractory)
    
    def test_zero_firing_rate(self):
        """Test handling of zero firing rate"""
        spike_times = self.gen.generate_spike_train(1.0, 0.0)
        assert len(spike_times) == 0
    
    # ===== Spike Injection Tests =====
    
    def test_spike_injection_count(self):
        """Test correct number of spikes are injected"""
        base_signal = np.zeros(20000)  # 1 second at 20 kHz
        spike_waveform = self.gen.generate_spike_waveform('biphasic')
        spike_times = np.array([0.1, 0.3, 0.5, 0.7])
        
        result = self.gen.inject_spikes(base_signal, spike_times, spike_waveform)
        
        # Signal should be modified
        assert not np.allclose(result, base_signal)
        
        # Check spikes were added at approximately correct locations
        for spike_time in spike_times:
            idx = int(spike_time * self.gen.fs)
            if idx < len(result) - len(spike_waveform):
                # Signal should have non-zero values around spike time
                spike_region = result[idx:idx+len(spike_waveform)]
                assert np.sum(np.abs(spike_region)) > 0
    
    def test_spike_injection_preserves_base(self):
        """Test spike injection preserves base signal characteristics"""
        base_signal = np.ones(20000) * 0.5  # Constant baseline
        spike_waveform = self.gen.generate_spike_waveform('simple')
        spike_times = np.array([0.1, 0.5])
        
        result = self.gen.inject_spikes(base_signal, spike_times, spike_waveform)
        
        # Baseline should be preserved in non-spike regions
        # Check a region without spikes
        no_spike_region = result[1000:1500]
        assert np.mean(no_spike_region) > 0.4  # Close to 0.5
    
    def test_spike_injection_boundary(self):
        """Test spike injection handles boundaries correctly"""
        base_signal = np.zeros(1000)
        spike_waveform = self.gen.generate_spike_waveform('biphasic')
        
        # Spike near end of signal (should be skipped)
        spike_times = np.array([0.049])  # Very close to end
        
        result = self.gen.inject_spikes(base_signal, spike_times, spike_waveform)
        
        # Should not crash
        assert len(result) == len(base_signal)
    
    # ===== Complete Signal Generation Tests =====
    
    def test_generate_signal_output_shapes(self):
        """Test generate_signal returns correct shapes"""
        duration = 1.0
        t, signal, spike_times, spike_indices = self.gen.generate_signal(
            duration=duration, firing_rate=30.0
        )
        
        expected_length = int(duration * self.gen.fs)
        assert len(t) == expected_length
        assert len(signal) == expected_length
        assert len(spike_times) > 0
        assert len(spike_indices) > 0
    
    def test_generate_signal_reproducibility(self):
        """Test signal generation is reproducible with same seed"""
        duration = 1.0
        
        t1, signal1, spikes1, indices1 = self.gen.generate_signal(
            duration=duration, firing_rate=30.0, seed=42
        )
        
        t2, signal2, spikes2, indices2 = self.gen.generate_signal(
            duration=duration, firing_rate=30.0, seed=42
        )
        
        assert np.allclose(signal1, signal2)
        assert np.allclose(spikes1, spikes2)
    
    def test_generate_signal_components(self):
        """Test generated signal contains all components"""
        duration = 1.0
        t, signal, spike_times, spike_indices = self.gen.generate_signal(
            duration=duration, firing_rate=30.0, seed=42
        )
        
        # Signal should have non-zero variance (noise + drift + spikes)
        assert np.std(signal) > 0.01
        
        # Should have spikes
        assert len(spike_times) > 10  # Expect ~30 spikes
        
        # Signal should have values in reasonable range
        assert np.max(np.abs(signal)) < 5.0
    
    def test_different_spike_types(self):
        """Test signal generation with different spike types"""
        for spike_type in ['biphasic', 'triphasic', 'simple']:
            t, signal, spikes, indices = self.gen.generate_signal(
                duration=0.5, 
                firing_rate=20.0, 
                spike_type=spike_type,
                seed=42
            )
            assert len(signal) > 0
            assert len(spikes) > 0
    
    # ===== Multi-Unit Signal Tests =====
    
    def test_multi_unit_signal_output(self):
        """Test multi-unit signal generation"""
        duration = 1.0
        num_units = 3
        
        t, signal, all_spike_times = self.gen.generate_multi_unit_signal(
            duration=duration,
            num_units=num_units,
            firing_rates=[20.0, 30.0, 40.0],
            seed=42
        )
        
        assert len(t) == int(duration * self.gen.fs)
        assert len(signal) == len(t)
        assert len(all_spike_times) == num_units
    
    def test_multi_unit_default_rates(self):
        """Test multi-unit with default firing rates"""
        t, signal, all_spike_times = self.gen.generate_multi_unit_signal(
            duration=0.5,
            num_units=2,
            seed=42
        )
        
        assert len(all_spike_times) == 2
        # Each unit should have some spikes
        for spike_times in all_spike_times:
            assert len(spike_times) > 0
    
    # ===== CSV Export Tests =====
    
    def test_export_to_csv(self):
        """Test CSV export functionality"""
        import tempfile
        import os
        
        # Generate signal
        t, signal, spike_times, spike_indices = self.gen.generate_signal(
            duration=0.1, firing_rate=20.0, seed=42
        )
        
        # Export to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            exported = self.gen.export_to_csv(
                t, signal, spike_times, spike_indices,
                output_dir=tmpdir,
                prefix='test'
            )
            
            # Check files were created
            assert 'signal_data' in exported
            assert 'summary' in exported
            assert os.path.exists(exported['signal_data'])
            assert os.path.exists(exported['summary'])
            
            # Check signal data file has content
            data = np.loadtxt(exported['signal_data'], delimiter=',', skiprows=1)
            assert data.shape[0] == len(signal)
            assert data.shape[1] == 3  # time, amplitude, spike_marker
    
    def test_export_waveform_to_csv(self):
        """Test waveform export functionality"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.gen.export_waveform_to_csv(
                spike_type='biphasic',
                output_dir=tmpdir,
                filename='test_waveform.csv'
            )
            
            assert os.path.exists(path)
            
            # Load and verify
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            assert data.shape[1] == 2  # time, amplitude
            assert data.shape[0] > 0
    
    # ===== Edge Cases and Error Handling =====
    
    def test_very_short_duration(self):
        """Test with very short duration"""
        t, signal, spikes, indices = self.gen.generate_signal(
            duration=0.01, firing_rate=10.0
        )
        assert len(signal) > 0
    
    def test_very_long_duration(self):
        """Test with long duration (memory test)"""
        # Use lower sampling rate for this test
        gen = NeuralSignalGenerator(fs=1000)
        t, signal, spikes, indices = gen.generate_signal(
            duration=10.0, firing_rate=20.0
        )
        assert len(signal) == 10000
    
    def test_high_firing_rate(self):
        """Test with very high firing rate"""
        t, signal, spikes, indices = self.gen.generate_signal(
            duration=0.5, firing_rate=200.0
        )
        # Should still respect refractory period
        if len(spikes) > 1:
            isis = np.diff(spikes)
            assert np.all(isis >= 0.002)  # 2 ms refractory


def run_validation_tests():
    """
    Run validation tests and print results
    """
    print("=" * 70)
    print(" Signal Generator Validation Tests")
    print("=" * 70)
    
    # Create test instance
    test = TestNeuralSignalGenerator()
    test.setup_method()
    
    # List of all test methods
    test_methods = [
        method for method in dir(test) 
        if method.startswith('test_') and callable(getattr(test, method))
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
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
    # Run with pytest if available, otherwise use simple runner
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        success = run_validation_tests()
        sys.exit(0 if success else 1)

