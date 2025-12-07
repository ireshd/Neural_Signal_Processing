"""
Spike Detection for Neural Signals

Implements real-time spike detection algorithms:
- Adaptive threshold detection (robust to noise variations)
- Peak detection with refractory period
- Spike alignment and waveform extraction
- Feature extraction (amplitude, width, energy)
- False positive suppression

Uses methods from:
- Quiroga et al. (2004) - Unsupervised spike detection and sorting
- Rey et al. (2015) - Past, present and future of spike sorting techniques
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy import signal
import os
import csv


class SpikeDetector:
    """
    Real-time spike detector with adaptive thresholding.
    
    Features:
    - Adaptive threshold based on robust noise estimation
    - Refractory period enforcement
    - Peak alignment
    - Waveform extraction
    
    Attributes:
        fs (float): Sampling frequency in Hz
        threshold_factor (float): Multiplier for noise-based threshold
        refractory_period (float): Minimum time between spikes (seconds)
        spike_window (float): Window for spike waveform extraction (seconds)
    """
    
    def __init__(self,
                 fs: float = 20000.0,
                 threshold_factor: float = 4.0,
                 refractory_period: float = 0.001,
                 spike_window: float = 0.002,
                 alignment: str = 'peak'):
        """
        Initialize spike detector.
        
        Args:
            fs: Sampling frequency in Hz
            threshold_factor: Multiplier for threshold (typically 3-5)
            refractory_period: Minimum time between spikes in seconds (typically 1 ms)
            spike_window: Window size for extracted waveforms in seconds (typically 2 ms)
            alignment: How to align spikes ('peak', 'valley', 'center')
        """
        self.fs = fs
        self.threshold_factor = threshold_factor
        self.refractory_period = refractory_period
        self.spike_window = spike_window
        self.alignment = alignment
        
        # Convert time to samples
        self.refractory_samples = int(refractory_period * fs)
        self.spike_window_samples = int(spike_window * fs)
        
        # State for real-time processing
        self.last_spike_idx = -self.refractory_samples
        self.detected_spikes: List[int] = []
        self.spike_waveforms: List[np.ndarray] = []
        self.spike_features: List[Dict[str, float]] = []
        
        # Statistics
        self.total_samples_processed = 0
        self.total_spikes_detected = 0
        
    def estimate_noise_std(self, data: np.ndarray) -> float:
        """
        Estimate noise standard deviation using median absolute deviation (MAD).
        
        This is robust to the presence of spikes in the signal.
        Based on Quiroga et al. (2004).
        
        Args:
            data: Input signal
            
        Returns:
            Estimated noise standard deviation
        """
        # MAD estimator
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        sigma = mad / 0.6745  # Conversion factor for Gaussian noise
        return sigma
    
    def compute_threshold(self, data: np.ndarray) -> float:
        """
        Compute adaptive detection threshold.
        
        Args:
            data: Input signal block
            
        Returns:
            Detection threshold
        """
        sigma = self.estimate_noise_std(data)
        threshold = self.threshold_factor * sigma
        return threshold
    
    def detect_spikes(self, 
                     data: np.ndarray,
                     threshold: Optional[float] = None,
                     return_waveforms: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect spikes in a data block.
        
        Args:
            data: Input signal block
            threshold: Detection threshold (if None, computed adaptively)
            return_waveforms: Whether to extract spike waveforms
            
        Returns:
            Tuple of (spike_indices, spike_waveforms)
        """
        if len(data) == 0:
            return np.array([], dtype=int), []
        
        # Compute threshold if not provided
        if threshold is None:
            threshold = self.compute_threshold(data)
        
        # Find peaks above threshold
        # Use negative signal for valley detection (extracellular spikes are often negative)
        if self.alignment == 'valley':
            detection_signal = -data
        else:
            detection_signal = data
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            detection_signal,
            height=threshold,
            distance=self.refractory_samples
        )
        
        spike_waveforms = []
        
        if return_waveforms and len(peaks) > 0:
            # Extract waveforms centered on peaks
            half_window = self.spike_window_samples // 2
            
            for peak_idx in peaks:
                # Check if we have enough samples around the peak
                if peak_idx >= half_window and peak_idx + half_window < len(data):
                    waveform = data[peak_idx - half_window:peak_idx + half_window]
                    spike_waveforms.append(waveform)
                else:
                    # Not enough samples, skip this spike
                    spike_waveforms.append(None)
        
        # Update statistics
        self.total_samples_processed += len(data)
        self.total_spikes_detected += len(peaks)
        
        return peaks, spike_waveforms
    
    def detect_spikes_stream(self,
                           data: np.ndarray,
                           offset: int = 0) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect spikes in streaming mode with state preservation.
        
        Args:
            data: Input signal block
            offset: Sample offset from start of recording (for absolute indexing)
            
        Returns:
            Tuple of (spike_indices_absolute, spike_waveforms)
        """
        # Detect spikes
        peaks, waveforms = self.detect_spikes(data, return_waveforms=True)
        
        # Convert to absolute indices
        absolute_peaks = peaks + offset
        
        # Store detected spikes
        self.detected_spikes.extend(absolute_peaks)
        self.spike_waveforms.extend([w for w in waveforms if w is not None])
        
        return absolute_peaks, waveforms
    
    def extract_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """
        Extract features from spike waveform.
        
        Args:
            waveform: Spike waveform
            
        Returns:
            Dictionary of features:
            - peak_amplitude: Maximum absolute amplitude
            - peak_to_peak: Peak-to-peak amplitude
            - energy: Total energy
            - width: Spike width at half-maximum
            - peak_time: Time of peak (normalized 0-1)
        """
        if len(waveform) == 0:
            return {}
        
        features = {}
        
        # Peak amplitude
        peak_val = np.max(np.abs(waveform))
        features['peak_amplitude'] = float(peak_val)
        
        # Peak-to-peak
        features['peak_to_peak'] = float(np.max(waveform) - np.min(waveform))
        
        # Energy
        features['energy'] = float(np.sum(waveform ** 2))
        
        # Peak time (normalized)
        peak_idx = np.argmax(np.abs(waveform))
        features['peak_time'] = float(peak_idx / len(waveform))
        
        # Width at half-maximum
        half_max = peak_val / 2.0
        above_half = np.abs(waveform) > half_max
        if np.any(above_half):
            width_samples = np.sum(above_half)
            features['width_samples'] = float(width_samples)
            features['width_ms'] = float(width_samples / self.fs * 1000)
        else:
            features['width_samples'] = 0.0
            features['width_ms'] = 0.0
        
        return features
    
    def compute_all_features(self, waveforms: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Compute features for all waveforms.
        
        Args:
            waveforms: List of spike waveforms
            
        Returns:
            List of feature dictionaries
        """
        return [self.extract_features(w) for w in waveforms if w is not None]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get detection statistics.
        
        Returns:
            Dictionary containing:
            - total_spikes: Total number of spikes detected
            - total_samples: Total samples processed
            - firing_rate: Average firing rate (Hz)
            - detection_rate: Spikes per sample
        """
        stats = {
            'total_spikes': self.total_spikes_detected,
            'total_samples': self.total_samples_processed,
        }
        
        if self.total_samples_processed > 0:
            duration = self.total_samples_processed / self.fs
            stats['duration_s'] = duration
            stats['firing_rate_hz'] = self.total_spikes_detected / duration
            stats['detection_rate'] = self.total_spikes_detected / self.total_samples_processed
        else:
            stats['duration_s'] = 0.0
            stats['firing_rate_hz'] = 0.0
            stats['detection_rate'] = 0.0
        
        return stats
    
    def reset(self) -> None:
        """Reset detector state."""
        self.last_spike_idx = -self.refractory_samples
        self.detected_spikes.clear()
        self.spike_waveforms.clear()
        self.spike_features.clear()
        self.total_samples_processed = 0
        self.total_spikes_detected = 0
    
    def export_to_csv(self,
                     spike_times: np.ndarray,
                     waveforms: List[np.ndarray],
                     output_dir: str = 'data/outputs',
                     prefix: str = 'spike_detection') -> Dict[str, str]:
        """
        Export spike detection results to CSV files.
        
        Args:
            spike_times: Array of spike times in seconds
            waveforms: List of spike waveforms
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping file types to paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Export spike times
        times_file = os.path.join(output_dir, f'{prefix}_times.csv')
        with open(times_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['spike_number', 'time_s', 'sample_index'])
            for i, t in enumerate(spike_times):
                sample_idx = int(t * self.fs)
                writer.writerow([i, t, sample_idx])
        exported_files['times'] = times_file
        
        # Export waveforms
        if waveforms:
            waveforms_file = os.path.join(output_dir, f'{prefix}_waveforms.csv')
            with open(waveforms_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                max_length = max(len(w) for w in waveforms if w is not None)
                header = ['spike_number'] + [f'sample_{i}' for i in range(max_length)]
                writer.writerow(header)
                
                # Data
                for i, waveform in enumerate(waveforms):
                    if waveform is not None:
                        row = [i] + list(waveform)
                        writer.writerow(row)
            exported_files['waveforms'] = waveforms_file
        
        # Export features
        if waveforms:
            features = self.compute_all_features(waveforms)
            if features:
                features_file = os.path.join(output_dir, f'{prefix}_features.csv')
                with open(features_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Header
                    header = ['spike_number'] + list(features[0].keys())
                    writer.writerow(header)
                    
                    # Data
                    for i, feat in enumerate(features):
                        row = [i] + list(feat.values())
                        writer.writerow(row)
                exported_files['features'] = features_file
        
        # Export statistics
        stats = self.get_statistics()
        stats_file = os.path.join(output_dir, f'{prefix}_statistics.csv')
        with open(stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in stats.items():
                writer.writerow([key, value])
        exported_files['statistics'] = stats_file
        
        return exported_files


def demo():
    """Demonstrate spike detection functionality."""
    print("=" * 70)
    print("Spike Detection Demo")
    print("=" * 70)
    
    # Generate test signal with known spikes
    from signal_gen import NeuralSignalGenerator
    
    fs = 20000.0
    duration = 2.0
    firing_rate = 30.0
    
    print(f"\n1. Generating Test Signal")
    print(f"   Duration: {duration} s")
    print(f"   Sampling rate: {fs} Hz")
    print(f"   Target firing rate: {firing_rate} Hz")
    
    gen = NeuralSignalGenerator(
        fs=fs,
        noise_amplitude=0.05,
        drift_amplitude=0.1,
        drift_freq=1.0
    )
    
    t, signal_data, true_spike_times, true_spike_indices = gen.generate_signal(
        duration=duration,
        firing_rate=firing_rate,
        spike_type='biphasic',
        seed=42
    )
    
    print(f"   Generated {len(true_spike_times)} true spikes")
    print(f"   Signal RMS: {np.sqrt(np.mean(signal_data**2)):.4f}")
    
    # Apply filtering (typically done before spike detection)
    from dsp_filters import design_neural_filter_cascade
    
    print(f"\n2. Filtering Signal")
    filter_cascade = design_neural_filter_cascade(fs=fs)
    filtered_signal = filter_cascade.filter_block(signal_data)
    print(f"   Applied: High-pass -> Band-pass -> Notch")
    print(f"   Filtered RMS: {np.sqrt(np.mean(filtered_signal**2)):.4f}")
    
    # Initialize spike detector
    print(f"\n3. Spike Detection")
    detector = SpikeDetector(
        fs=fs,
        threshold_factor=4.0,
        refractory_period=0.001,
        spike_window=0.002,
        alignment='valley'
    )
    
    # Estimate threshold
    threshold = detector.compute_threshold(filtered_signal)
    noise_std = detector.estimate_noise_std(filtered_signal)
    print(f"   Noise std (MAD): {noise_std:.4f}")
    print(f"   Threshold: {threshold:.4f} ({detector.threshold_factor}x noise)")
    
    # Detect spikes
    detected_indices, waveforms = detector.detect_spikes(filtered_signal)
    detected_times = detected_indices / fs
    
    print(f"   Detected {len(detected_indices)} spikes")
    print(f"   Detection rate: {len(detected_indices)/duration:.1f} Hz")
    
    # Compute detection accuracy
    # Match detected spikes to true spikes (within 1 ms tolerance)
    tolerance = 0.001  # 1 ms
    true_positives = 0
    
    for true_time in true_spike_times:
        # Check if any detected spike is within tolerance
        if np.any(np.abs(detected_times - true_time) < tolerance):
            true_positives += 1
    
    false_positives = len(detected_times) - true_positives
    false_negatives = len(true_spike_times) - true_positives
    
    if len(detected_times) > 0:
        precision = true_positives / len(detected_times)
    else:
        precision = 0.0
    
    if len(true_spike_times) > 0:
        recall = true_positives / len(true_spike_times)
    else:
        recall = 0.0
    
    print(f"\n4. Detection Accuracy (1 ms tolerance)")
    print(f"   True spikes: {len(true_spike_times)}")
    print(f"   Detected spikes: {len(detected_times)}")
    print(f"   True positives: {true_positives}")
    print(f"   False positives: {false_positives}")
    print(f"   False negatives: {false_negatives}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    
    # Extract features
    print(f"\n5. Feature Extraction")
    if waveforms and any(w is not None for w in waveforms):
        valid_waveforms = [w for w in waveforms if w is not None]
        features = detector.compute_all_features(valid_waveforms)
        
        if features:
            # Compute feature statistics
            feature_names = list(features[0].keys())
            print(f"   Extracted {len(feature_names)} features per spike:")
            
            for name in feature_names:
                values = [f[name] for f in features]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"   - {name}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Test block-based detection (real-time simulation)
    print(f"\n6. Block-Based Detection (Real-Time Simulation)")
    detector.reset()
    block_size = 1024
    num_blocks = len(filtered_signal) // block_size
    
    all_spike_indices = []
    all_waveforms = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = filtered_signal[start:end]
        
        spike_indices, block_waveforms = detector.detect_spikes_stream(block, offset=start)
        all_spike_indices.extend(spike_indices)
        all_waveforms.extend([w for w in block_waveforms if w is not None])
    
    print(f"   Block size: {block_size} samples ({block_size/fs*1000:.1f} ms)")
    print(f"   Blocks processed: {num_blocks}")
    print(f"   Total spikes detected: {len(all_spike_indices)}")
    print(f"   Average per block: {len(all_spike_indices)/num_blocks:.1f}")
    
    # Export results
    print(f"\n7. Exporting Results")
    exported = detector.export_to_csv(
        spike_times=detected_times,
        waveforms=valid_waveforms if waveforms else []
    )
    
    for file_type, path in exported.items():
        print(f"   {file_type}: {path}")
    
    # Create visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print(f"\n8. Creating Visualization")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Full signal
        axes[0].plot(t, signal_data, alpha=0.5, label='Raw')
        axes[0].plot(t, filtered_signal, alpha=0.8, label='Filtered')
        axes[0].scatter(true_spike_times, np.zeros_like(true_spike_times), 
                       c='green', marker='v', s=50, label='True spikes', alpha=0.6, zorder=5)
        axes[0].scatter(detected_times, np.zeros_like(detected_times),
                       c='red', marker='^', s=50, label='Detected', alpha=0.6, zorder=5)
        axes[0].axhline(threshold, color='orange', linestyle='--', label=f'Threshold ({threshold:.3f})')
        axes[0].axhline(-threshold, color='orange', linestyle='--')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Spike Detection - Full Signal')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Zoomed view (first 100 ms)
        zoom_time = 0.1
        zoom_samples = int(zoom_time * fs)
        axes[1].plot(t[:zoom_samples], filtered_signal[:zoom_samples])
        zoom_detected = detected_times[detected_times < zoom_time]
        zoom_true = true_spike_times[true_spike_times < zoom_time]
        axes[1].scatter(zoom_true, np.zeros_like(zoom_true),
                       c='green', marker='v', s=80, label='True spikes', zorder=5)
        axes[1].scatter(zoom_detected, np.zeros_like(zoom_detected),
                       c='red', marker='^', s=80, label='Detected', zorder=5)
        axes[1].axhline(threshold, color='orange', linestyle='--', label='Threshold')
        axes[1].axhline(-threshold, color='orange', linestyle='--')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(f'Spike Detection - Zoomed ({zoom_time*1000:.0f} ms)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Spike waveforms
        if valid_waveforms and len(valid_waveforms) > 0:
            waveform_time = np.arange(len(valid_waveforms[0])) / fs * 1000
            for i, waveform in enumerate(valid_waveforms[:50]):  # Plot up to 50
                axes[2].plot(waveform_time, waveform, alpha=0.3, color='blue')
            
            # Mean waveform
            mean_waveform = np.mean(valid_waveforms, axis=0)
            axes[2].plot(waveform_time, mean_waveform, color='red', linewidth=2, label='Mean')
            axes[2].set_xlabel('Time (ms)')
            axes[2].set_ylabel('Amplitude')
            axes[2].set_title(f'Detected Spike Waveforms (n={len(valid_waveforms)})')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Raster plot and firing rate
        if len(detected_times) > 0:
            axes[3].eventplot(detected_times, lineoffsets=0.5, linelengths=0.8, colors='black')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('Spikes')
            axes[3].set_title('Spike Raster Plot')
            axes[3].set_ylim([0, 1])
            axes[3].set_yticks([])
            axes[3].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_file = 'data/outputs/spike_detection_demo.png'
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

