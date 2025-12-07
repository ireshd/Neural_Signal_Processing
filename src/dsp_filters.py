"""
DSP Filters for Neural Signal Processing

Implements real-time digital filters commonly used in neural recording systems:
- Band-pass filter (300-3000 Hz) for spike extraction
- Notch filter (60 Hz) for power line noise removal
- RMS/energy tracking
- Block-based filtering with state preservation
- Filter design and visualization tools

These filters process neural signals in real-time blocks, maintaining filter state
between blocks to avoid discontinuities.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, Dict, List
import csv
import os


class DSPFilter:
    """
    Real-time DSP filter with state management for block-based processing.
    
    Supports band-pass, notch, and other IIR/FIR filter types with proper
    state handling for continuous streaming data.
    
    Attributes:
        b (np.ndarray): Numerator coefficients
        a (np.ndarray): Denominator coefficients
        zi (np.ndarray): Filter state
        fs (float): Sampling frequency
    """
    
    def __init__(self, b: np.ndarray, a: np.ndarray, fs: float):
        """
        Initialize filter with coefficients.
        
        Args:
            b: Numerator (feedforward) coefficients
            a: Denominator (feedback) coefficients
            fs: Sampling frequency in Hz
        """
        self.b = b
        self.a = a
        self.fs = fs
        self.zi = signal.lfilter_zi(b, a)
        self.reset_state()
        
        # Statistics
        self.blocks_processed = 0
        self.samples_processed = 0
        
    def filter_block(self, data: np.ndarray) -> np.ndarray:
        """
        Filter a block of data while preserving state.
        
        Args:
            data: Input data block
            
        Returns:
            Filtered data block
        """
        if len(data) == 0:
            return data
        
        # Apply filter with state
        filtered, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
        
        self.blocks_processed += 1
        self.samples_processed += len(data)
        
        return filtered
    
    def reset_state(self) -> None:
        """Reset filter state to initial conditions."""
        self.zi = signal.lfilter_zi(self.b, self.a) * 0.0
        self.blocks_processed = 0
        self.samples_processed = 0
    
    def get_frequency_response(self, num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get filter frequency response.
        
        Args:
            num_points: Number of frequency points
            
        Returns:
            Tuple of (frequencies in Hz, magnitude in dB)
        """
        w, h = signal.freqz(self.b, self.a, worN=num_points)
        freq = w * self.fs / (2 * np.pi)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        return freq, magnitude_db


class BandPassFilter(DSPFilter):
    """
    Band-pass filter for neural spike extraction.
    
    Typical settings: 300-3000 Hz for single-unit activity
    """
    
    def __init__(self, 
                 fs: float = 20000.0,
                 lowcut: float = 300.0,
                 highcut: float = 3000.0,
                 order: int = 4):
        """
        Design and initialize band-pass filter.
        
        Args:
            fs: Sampling frequency in Hz
            lowcut: Lower cutoff frequency in Hz
            highcut: Upper cutoff frequency in Hz
            order: Filter order (higher = sharper rolloff)
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        
        # Design Butterworth band-pass filter
        nyquist = fs / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        super().__init__(b, a, fs)


class NotchFilter(DSPFilter):
    """
    Notch filter for power line noise removal.
    
    Typical settings: 60 Hz (US) or 50 Hz (Europe) with Q=30
    """
    
    def __init__(self,
                 fs: float = 20000.0,
                 notch_freq: float = 60.0,
                 quality_factor: float = 30.0):
        """
        Design and initialize notch filter.
        
        Args:
            fs: Sampling frequency in Hz
            notch_freq: Frequency to notch out in Hz
            quality_factor: Q factor (higher = narrower notch)
        """
        self.notch_freq = notch_freq
        self.quality_factor = quality_factor
        
        # Design IIR notch filter
        nyquist = fs / 2.0
        w0 = notch_freq / nyquist
        
        b, a = signal.iirnotch(w0, quality_factor)
        
        super().__init__(b, a, fs)


class HighPassFilter(DSPFilter):
    """
    High-pass filter for DC offset and drift removal.
    
    Typical settings: 1-10 Hz cutoff
    """
    
    def __init__(self,
                 fs: float = 20000.0,
                 cutoff: float = 1.0,
                 order: int = 4):
        """
        Design and initialize high-pass filter.
        
        Args:
            fs: Sampling frequency in Hz
            cutoff: Cutoff frequency in Hz
            order: Filter order
        """
        self.cutoff = cutoff
        self.order = order
        
        # Design Butterworth high-pass filter
        nyquist = fs / 2.0
        wc = cutoff / nyquist
        
        b, a = signal.butter(order, wc, btype='high')
        
        super().__init__(b, a, fs)


class LowPassFilter(DSPFilter):
    """
    Low-pass filter for anti-aliasing or smoothing.
    
    Typical settings: 5000-8000 Hz for neural signals
    """
    
    def __init__(self,
                 fs: float = 20000.0,
                 cutoff: float = 5000.0,
                 order: int = 4):
        """
        Design and initialize low-pass filter.
        
        Args:
            fs: Sampling frequency in Hz
            cutoff: Cutoff frequency in Hz
            order: Filter order
        """
        self.cutoff = cutoff
        self.order = order
        
        # Design Butterworth low-pass filter
        nyquist = fs / 2.0
        wc = cutoff / nyquist
        
        b, a = signal.butter(order, wc, btype='low')
        
        super().__init__(b, a, fs)


class FilterCascade:
    """
    Cascade of multiple filters applied in sequence.
    
    Example: High-pass -> Band-pass -> Notch
    """
    
    def __init__(self, filters: List[DSPFilter]):
        """
        Initialize filter cascade.
        
        Args:
            filters: List of DSPFilter objects to apply in order
        """
        self.filters = filters
        
    def filter_block(self, data: np.ndarray) -> np.ndarray:
        """
        Apply all filters in cascade.
        
        Args:
            data: Input data block
            
        Returns:
            Filtered data after all stages
        """
        filtered = data.copy()
        for filt in self.filters:
            filtered = filt.filter_block(filtered)
        return filtered
    
    def reset_state(self) -> None:
        """Reset all filter states."""
        for filt in self.filters:
            filt.reset_state()


class RMSTracker:
    """
    Real-time RMS (Root Mean Square) energy tracker.
    
    Computes RMS energy in sliding windows for activity monitoring.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize RMS tracker.
        
        Args:
            window_size: Window size for RMS calculation (in samples)
        """
        self.window_size = window_size
        self.history = np.zeros(window_size)
        self.idx = 0
        self.filled = False
        
    def update(self, data: np.ndarray) -> np.ndarray:
        """
        Update RMS with new data block.
        
        Args:
            data: New data samples
            
        Returns:
            RMS value for each sample in data
        """
        rms_values = np.zeros(len(data))
        
        for i, sample in enumerate(data):
            # Add new sample
            self.history[self.idx] = sample
            self.idx += 1
            
            if self.idx >= self.window_size:
                self.idx = 0
                self.filled = True
            
            # Calculate RMS
            if self.filled:
                rms_values[i] = np.sqrt(np.mean(self.history ** 2))
            else:
                # Not enough samples yet
                rms_values[i] = np.sqrt(np.mean(self.history[:self.idx] ** 2))
        
        return rms_values
    
    def get_current_rms(self) -> float:
        """Get current RMS value."""
        if self.filled:
            return float(np.sqrt(np.mean(self.history ** 2)))
        elif self.idx > 0:
            return float(np.sqrt(np.mean(self.history[:self.idx] ** 2)))
        else:
            return 0.0
    
    def reset(self) -> None:
        """Reset RMS tracker."""
        self.history.fill(0)
        self.idx = 0
        self.filled = False


def design_neural_filter_cascade(fs: float = 20000.0) -> FilterCascade:
    """
    Design a standard neural signal processing filter cascade.
    
    Applies:
    1. High-pass (1 Hz) - Remove DC offset and slow drift
    2. Band-pass (300-3000 Hz) - Extract spike band
    3. Notch (60 Hz) - Remove power line noise
    
    Args:
        fs: Sampling frequency in Hz
        
    Returns:
        FilterCascade ready for processing
    """
    filters = [
        HighPassFilter(fs=fs, cutoff=1.0, order=4),
        BandPassFilter(fs=fs, lowcut=300.0, highcut=3000.0, order=4),
        NotchFilter(fs=fs, notch_freq=60.0, quality_factor=30.0)
    ]
    return FilterCascade(filters)


def compute_rms(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    Compute RMS energy in sliding window.
    
    Args:
        data: Input signal
        window_size: Window size in samples
        
    Returns:
        RMS values
    """
    tracker = RMSTracker(window_size=window_size)
    return tracker.update(data)


def export_filter_response_to_csv(filt: DSPFilter,
                                  output_dir: str = 'data/outputs',
                                  prefix: str = 'filter_response') -> str:
    """
    Export filter frequency response to CSV.
    
    Args:
        filt: DSPFilter object
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Path to CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    freq, magnitude = filt.get_frequency_response(num_points=1024)
    
    filepath = os.path.join(output_dir, f'{prefix}.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frequency_hz', 'magnitude_db'])
        for f_val, m_val in zip(freq, magnitude):
            writer.writerow([f_val, m_val])
    
    return filepath


def demo():
    """Demonstrate DSP filter functionality."""
    print("=" * 70)
    print("DSP Filters Demo")
    print("=" * 70)
    
    # Generate test signal with multiple frequency components
    fs = 20000.0
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create test signal:
    # - DC offset
    # - 1 Hz drift
    # - 60 Hz noise (power line)
    # - 500 Hz spike-like activity
    # - 2000 Hz spike-like activity
    # - Gaussian noise
    
    signal_test = (
        0.5 +  # DC offset
        0.3 * np.sin(2 * np.pi * 1.0 * t) +  # 1 Hz drift
        0.1 * np.sin(2 * np.pi * 60.0 * t) +  # 60 Hz noise
        0.2 * np.sin(2 * np.pi * 500.0 * t) +  # Spike band
        0.15 * np.sin(2 * np.pi * 2000.0 * t) +  # Spike band
        0.05 * np.random.randn(len(t))  # Noise
    )
    
    print(f"\n1. Test Signal")
    print(f"   Duration: {duration} s")
    print(f"   Sampling rate: {fs} Hz")
    print(f"   Samples: {len(signal_test)}")
    print(f"   Components: DC, 1Hz drift, 60Hz noise, 500Hz + 2kHz spikes, noise")
    
    # Test individual filters
    print(f"\n2. Individual Filters")
    
    # High-pass filter
    hpf = HighPassFilter(fs=fs, cutoff=1.0, order=4)
    signal_hp = hpf.filter_block(signal_test)
    print(f"   High-pass (1 Hz): Removes DC and drift")
    print(f"   Output RMS: {np.sqrt(np.mean(signal_hp**2)):.4f}")
    
    # Band-pass filter
    bpf = BandPassFilter(fs=fs, lowcut=300.0, highcut=3000.0, order=4)
    signal_bp = bpf.filter_block(signal_test)
    print(f"   Band-pass (300-3000 Hz): Extracts spike band")
    print(f"   Output RMS: {np.sqrt(np.mean(signal_bp**2)):.4f}")
    
    # Notch filter
    notch = NotchFilter(fs=fs, notch_freq=60.0, quality_factor=30.0)
    signal_notch = notch.filter_block(signal_test)
    print(f"   Notch (60 Hz): Removes power line noise")
    print(f"   Output RMS: {np.sqrt(np.mean(signal_notch**2)):.4f}")
    
    # Test filter cascade (standard neural pipeline)
    print(f"\n3. Filter Cascade (Standard Neural Pipeline)")
    cascade = design_neural_filter_cascade(fs=fs)
    signal_filtered = cascade.filter_block(signal_test)
    print(f"   Pipeline: High-pass -> Band-pass -> Notch")
    print(f"   Input RMS: {np.sqrt(np.mean(signal_test**2)):.4f}")
    print(f"   Output RMS: {np.sqrt(np.mean(signal_filtered**2)):.4f}")
    
    # Test block-based processing with state preservation
    print(f"\n4. Block-Based Processing (Real-Time Simulation)")
    block_size = 512
    num_blocks = len(signal_test) // block_size
    
    cascade.reset_state()
    signal_blocked = np.zeros_like(signal_test)
    
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = signal_test[start:end]
        filtered_block = cascade.filter_block(block)
        signal_blocked[start:end] = filtered_block
    
    print(f"   Block size: {block_size} samples")
    print(f"   Blocks processed: {num_blocks}")
    print(f"   Total samples: {num_blocks * block_size}")
    
    # Verify consistency between full and blocked processing
    difference = np.mean(np.abs(signal_filtered[:num_blocks*block_size] - signal_blocked[:num_blocks*block_size]))
    print(f"   Difference vs. full filtering: {difference:.6f} (should be ~0)")
    
    # Test RMS tracker
    print(f"\n5. RMS Energy Tracking")
    rms_tracker = RMSTracker(window_size=100)
    rms_values = rms_tracker.update(signal_filtered)
    print(f"   Window size: 100 samples (5 ms @ 20 kHz)")
    print(f"   Current RMS: {rms_tracker.get_current_rms():.4f}")
    print(f"   Mean RMS: {np.mean(rms_values):.4f}")
    print(f"   Max RMS: {np.max(rms_values):.4f}")
    
    # Export filter responses
    print(f"\n6. Exporting Filter Frequency Responses")
    
    filters_to_export = [
        (hpf, 'highpass_1hz'),
        (bpf, 'bandpass_300_3000hz'),
        (notch, 'notch_60hz')
    ]
    
    for filt, name in filters_to_export:
        filepath = export_filter_response_to_csv(filt, prefix=name)
        print(f"   {name}: {filepath}")
    
    # Create visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        print(f"\n7. Creating Visualization")
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Time domain signals
        plot_duration = 0.1  # Show 100 ms
        plot_samples = int(plot_duration * fs)
        
        # Original signal
        axes[0, 0].plot(t[:plot_samples] * 1000, signal_test[:plot_samples])
        axes[0, 0].set_title('Original Signal (100 ms)')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Filtered signal
        axes[0, 1].plot(t[:plot_samples] * 1000, signal_filtered[:plot_samples])
        axes[0, 1].set_title('Filtered Signal (100 ms)')
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency response (combined plot)
        ax_freq = axes[1, 0]
        for filt, name, label in [
            (hpf, 'highpass_1hz', 'High-pass (1 Hz)'),
            (bpf, 'bandpass_300_3000hz', 'Band-pass (300-3000 Hz)'),
            (notch, 'notch_60hz', 'Notch (60 Hz)')
        ]:
            freq, mag = filt.get_frequency_response()
            ax_freq.plot(freq, mag, label=label, linewidth=2)
        
        ax_freq.set_title('Filter Frequency Responses')
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.set_ylabel('Magnitude (dB)')
        ax_freq.legend()
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xscale('log')
        
        # RMS tracking
        axes[1, 1].plot(t[:len(rms_values)] * 1000, rms_values)
        axes[1, 1].set_title('RMS Energy Tracking')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('RMS')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Power spectral density
        ax_psd = axes[2, 0]
        from scipy.signal import welch
        f_orig, Pxx_orig = welch(signal_test, fs=fs, nperseg=512)
        f_filt, Pxx_filt = welch(signal_filtered, fs=fs, nperseg=512)
        ax_psd.semilogy(f_orig, Pxx_orig, label='Original', alpha=0.7)
        ax_psd.semilogy(f_filt, Pxx_filt, label='Filtered', alpha=0.7)
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power (V²/Hz)')
        ax_psd.set_title('Power Spectral Density')
        ax_psd.legend()
        ax_psd.grid(True, alpha=0.3)
        ax_psd.set_xlim([0, 5000])
        
        # Comparison
        axes[2, 1].plot(t[:plot_samples] * 1000, signal_test[:plot_samples], 
                       alpha=0.5, label='Original')
        axes[2, 1].plot(t[:plot_samples] * 1000, signal_filtered[:plot_samples],
                       alpha=0.8, label='Filtered')
        axes[2, 1].set_title('Original vs Filtered (Overlay)')
        axes[2, 1].set_xlabel('Time (ms)')
        axes[2, 1].set_ylabel('Amplitude')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = 'data/outputs/dsp_filters_demo.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Visualization saved: {output_file}")
        
    except ImportError:
        print(f"   Matplotlib not available, skipping visualization")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()

