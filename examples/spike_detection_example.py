"""
Spike Detection Example

Demonstrates spike detection algorithms and feature extraction:
- Adaptive threshold detection
- Spike waveform extraction
- Feature computation
- Detection accuracy evaluation
- Real-time streaming detection
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_gen import NeuralSignalGenerator
from dsp_filters import design_neural_filter_cascade
from spike_detect import SpikeDetector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def example1_basic_detection():
    """Example 1: Basic spike detection."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Spike Detection")
    print("=" * 70)
    
    fs = 20000.0
    
    # Generate signal
    gen = NeuralSignalGenerator(fs=fs)
    t, signal, true_spike_times, _ = gen.generate_signal(
        duration=2.0,
        firing_rate=30.0,
        spike_type='biphasic',
        seed=42
    )
    
    print(f"Generated signal: {len(signal)} samples")
    print(f"True spikes: {len(true_spike_times)}")
    
    # Filter
    filters = design_neural_filter_cascade(fs=fs)
    filtered = filters.filter_block(signal)
    
    # Detect
    detector = SpikeDetector(fs=fs, threshold_factor=4.0)
    spike_indices, waveforms = detector.detect_spikes(filtered)
    detected_times = spike_indices / fs
    
    threshold = detector.compute_threshold(filtered)
    noise_std = detector.estimate_noise_std(filtered)
    
    print(f"\nNoise estimation:")
    print(f"  Noise std: {noise_std:.4f}")
    print(f"  Threshold: {threshold:.4f} ({detector.threshold_factor}x)")
    
    print(f"\nDetection results:")
    print(f"  Detected: {len(detected_times)} spikes")
    print(f"  Waveforms: {sum(w is not None for w in waveforms)}")
    
    # Accuracy
    tolerance = 0.001
    tp = sum(any(abs(dt - tt) < tolerance for tt in true_spike_times) for dt in detected_times)
    fp = len(detected_times) - tp
    fn = len(true_spike_times) - tp
    
    print(f"\nAccuracy (1 ms tolerance):")
    print(f"  True positives: {tp}")
    print(f"  False positives: {fp}")
    print(f"  False negatives: {fn}")
    print(f"  Precision: {tp/len(detected_times):.1%}" if len(detected_times) > 0 else "N/A")
    print(f"  Recall: {tp/len(true_spike_times):.1%}" if len(true_spike_times) > 0 else "N/A")


def example2_feature_extraction():
    """Example 2: Spike feature extraction."""
    print("\n" + "=" * 70)
    print("Example 2: Spike Feature Extraction")
    print("=" * 70)
    
    fs = 20000.0
    
    # Generate signal with different spike types
    gen = NeuralSignalGenerator(fs=fs)
    
    spike_types = ['biphasic', 'triphasic', 'simple']
    all_features = []
    
    for spike_type in spike_types:
        t, signal, _, _ = gen.generate_signal(
            duration=1.0,
            firing_rate=20.0,
            spike_type=spike_type,
            seed=42
        )
        
        # Filter and detect
        filters = design_neural_filter_cascade(fs=fs)
        filtered = filters.filter_block(signal)
        
        detector = SpikeDetector(fs=fs)
        _, waveforms = detector.detect_spikes(filtered)
        
        valid_waveforms = [w for w in waveforms if w is not None]
        
        if valid_waveforms:
            features = detector.compute_all_features(valid_waveforms)
            all_features.append((spike_type, features))
            
            print(f"\n{spike_type.capitalize()} spikes ({len(features)} detected):")
            if features:
                for key in features[0].keys():
                    values = [f[key] for f in features]
                    print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    print("\n✓ Feature extraction complete")


def example3_parameter_comparison():
    """Example 3: Compare different threshold factors."""
    print("\n" + "=" * 70)
    print("Example 3: Threshold Parameter Comparison")
    print("=" * 70)
    
    fs = 20000.0
    
    # Generate signal
    gen = NeuralSignalGenerator(fs=fs)
    t, signal, true_spike_times, _ = gen.generate_signal(
        duration=3.0,
        firing_rate=40.0,
        seed=42
    )
    
    # Filter
    filters = design_neural_filter_cascade(fs=fs)
    filtered = filters.filter_block(signal)
    
    # Test different thresholds
    thresholds = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    print(f"\nTrue spikes: {len(true_spike_times)}")
    print("\nThreshold | Detected | Precision | Recall")
    print("-" * 50)
    
    tolerance = 0.001
    
    for th in thresholds:
        detector = SpikeDetector(fs=fs, threshold_factor=th)
        spike_indices, _ = detector.detect_spikes(filtered)
        detected_times = spike_indices / fs
        
        tp = sum(any(abs(dt - tt) < tolerance for tt in true_spike_times) for dt in detected_times)
        precision = tp / len(detected_times) if len(detected_times) > 0 else 0
        recall = tp / len(true_spike_times) if len(true_spike_times) > 0 else 0
        
        print(f"{th:6.1f}x   | {len(detected_times):8} | {precision:8.1%} | {recall:6.1%}")
    
    print("\n✓ Parameter comparison complete")


def example4_streaming_detection():
    """Example 4: Real-time streaming detection."""
    print("\n" + "=" * 70)
    print("Example 4: Real-Time Streaming Detection")
    print("=" * 70)
    
    fs = 20000.0
    block_size = 512
    duration = 5.0
    
    # Generate signal
    gen = NeuralSignalGenerator(fs=fs)
    t, signal, true_spike_times, _ = gen.generate_signal(
        duration=duration,
        firing_rate=30.0,
        seed=42
    )
    
    # Filter
    filters = design_neural_filter_cascade(fs=fs)
    filtered = filters.filter_block(signal)
    
    # Stream processing
    detector = SpikeDetector(fs=fs)
    num_blocks = len(filtered) // block_size
    
    print(f"Signal: {len(filtered)} samples")
    print(f"Block size: {block_size} ({block_size/fs*1000:.1f} ms)")
    print(f"Blocks: {num_blocks}")
    print("\nProcessing blocks...")
    
    all_spikes = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = filtered[start:end]
        
        spike_indices, _ = detector.detect_spikes_stream(block, offset=start)
        all_spikes.extend(spike_indices)
        
        if i % 20 == 0:
            print(f"  Block {i+1}/{num_blocks}: {len(spike_indices)} spikes detected")
    
    print(f"\nTotal detected: {len(all_spikes)} spikes")
    print(f"Average rate: {len(all_spikes)/duration:.1f} Hz")
    
    # Export results
    detected_times = np.array(all_spikes) / fs
    detector.export_to_csv(
        spike_times=detected_times,
        waveforms=detector.spike_waveforms,
        prefix='streaming_detection_example'
    )
    
    print("✓ Results exported to CSV")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SPIKE DETECTION EXAMPLES")
    print("=" * 70)
    
    example1_basic_detection()
    example2_feature_extraction()
    example3_parameter_comparison()
    example4_streaming_detection()
    
    print("\n" + "=" * 70)
    print("All spike detection examples complete!")
    print("Check data/outputs/ for results.")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()

