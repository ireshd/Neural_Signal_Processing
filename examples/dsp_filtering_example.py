"""
DSP Filtering Example

Demonstrates digital signal processing filters for neural signal cleaning:
- High-pass filter for DC removal
- Band-pass filter for spike extraction
- Notch filter for power line noise removal
- Filter cascades
- Real-time block processing
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_gen import NeuralSignalGenerator
from dsp_filters import (
    HighPassFilter, BandPassFilter, NotchFilter, LowPassFilter,
    FilterCascade, design_neural_filter_cascade, RMSTracker,
    export_filter_response_to_csv
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def example1_individual_filters():
    """Example 1: Individual filter types."""
    print("\n" + "=" * 70)
    print("Example 1: Individual Filter Types")
    print("=" * 70)
    
    fs = 20000.0
    duration = 1.0
    
    # Generate test signal with multiple frequency components
    t = np.linspace(0, duration, int(fs * duration))
    signal = (
        0.5 +  # DC offset
        0.3 * np.sin(2 * np.pi * 0.5 * t) +  # 0.5 Hz drift
        0.1 * np.sin(2 * np.pi * 60.0 * t) +  # 60 Hz noise
        0.2 * np.sin(2 * np.pi * 800.0 * t) +  # Spike band
        0.05 * np.random.randn(len(t))  # Noise
    )
    
    print(f"Generated test signal: {len(t)} samples, {duration} s")
    print(f"Components: DC, 0.5Hz drift, 60Hz noise, 800Hz spike activity, noise")
    
    # Apply different filters
    hpf = HighPassFilter(fs=fs, cutoff=1.0)
    bpf = BandPassFilter(fs=fs, lowcut=300.0, highcut=3000.0)
    notch = NotchFilter(fs=fs, notch_freq=60.0)
    lpf = LowPassFilter(fs=fs, cutoff=5000.0)
    
    signal_hp = hpf.filter_block(signal)
    signal_bp = bpf.filter_block(signal)
    signal_notch = notch.filter_block(signal)
    signal_lp = lpf.filter_block(signal)
    
    print("\nFilter Results:")
    print(f"  Original RMS: {np.sqrt(np.mean(signal**2)):.4f}")
    print(f"  High-pass RMS: {np.sqrt(np.mean(signal_hp**2)):.4f}")
    print(f"  Band-pass RMS: {np.sqrt(np.mean(signal_bp**2)):.4f}")
    print(f"  Notch RMS: {np.sqrt(np.mean(signal_notch**2)):.4f}")
    print(f"  Low-pass RMS: {np.sqrt(np.mean(signal_lp**2)):.4f}")
    
    # Export frequency responses
    print("\nExporting filter responses:")
    export_filter_response_to_csv(hpf, prefix='example_highpass')
    export_filter_response_to_csv(bpf, prefix='example_bandpass')
    export_filter_response_to_csv(notch, prefix='example_notch')
    export_filter_response_to_csv(lpf, prefix='example_lowpass')
    print("  ✓ All frequency responses exported")


def example2_filter_cascade():
    """Example 2: Filter cascade for neural signal processing."""
    print("\n" + "=" * 70)
    print("Example 2: Filter Cascade (Standard Neural Pipeline)")
    print("=" * 70)
    
    fs = 20000.0
    
    # Generate realistic neural signal
    gen = NeuralSignalGenerator(fs=fs)
    t, signal, spike_times, _ = gen.generate_signal(
        duration=2.0,
        firing_rate=40.0,
        spike_type='biphasic'
    )
    
    print(f"Generated neural signal: {len(signal)} samples")
    print(f"True spikes: {len(spike_times)}")
    
    # Apply standard filter cascade
    cascade = design_neural_filter_cascade(fs=fs)
    filtered = cascade.filter_block(signal)
    
    print(f"\nApplied filter cascade:")
    print(f"  1. High-pass (1 Hz)")
    print(f"  2. Band-pass (300-3000 Hz)")
    print(f"  3. Notch (60 Hz)")
    print(f"\nOriginal RMS: {np.sqrt(np.mean(signal**2)):.4f}")
    print(f"Filtered RMS: {np.sqrt(np.mean(filtered**2)):.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    plot_duration = 0.1  # 100 ms
    plot_samples = int(plot_duration * fs)
    
    axes[0].plot(t[:plot_samples], signal[:plot_samples], linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t[:plot_samples], filtered[:plot_samples], linewidth=0.5, color='green')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Filtered Signal (Cascade)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/outputs/filter_cascade_example.png', dpi=150)
    plt.close()
    
    print("✓ Visualization saved: filter_cascade_example.png")


def example3_realtime_processing():
    """Example 3: Real-time block-based processing."""
    print("\n" + "=" * 70)
    print("Example 3: Real-Time Block Processing")
    print("=" * 70)
    
    fs = 20000.0
    block_size = 512
    duration = 5.0
    
    # Generate signal
    gen = NeuralSignalGenerator(fs=fs)
    t, signal, _, _ = gen.generate_signal(duration=duration, firing_rate=30.0)
    
    num_blocks = len(signal) // block_size
    
    print(f"Signal: {len(signal)} samples, {duration} s")
    print(f"Block size: {block_size} samples ({block_size/fs*1000:.1f} ms)")
    print(f"Number of blocks: {num_blocks}")
    
    # Process block by block
    cascade = design_neural_filter_cascade(fs=fs)
    rms_tracker = RMSTracker(window_size=100)
    
    filtered_blocks = []
    rms_values = []
    
    import time
    processing_times = []
    
    for i in range(num_blocks):
        start_time = time.time()
        
        # Extract block
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block = signal[start_idx:end_idx]
        
        # Filter
        filtered_block = cascade.filter_block(block)
        filtered_blocks.append(filtered_block)
        
        # Track RMS
        block_rms = rms_tracker.update(filtered_block)
        rms_values.extend(block_rms)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        if i % 20 == 0:
            print(f"  Block {i+1}/{num_blocks}: {processing_time*1000:.3f} ms")
    
    filtered_signal = np.concatenate(filtered_blocks)
    
    print(f"\nProcessing Statistics:")
    print(f"  Mean time: {np.mean(processing_times)*1000:.3f} ms")
    print(f"  Max time: {np.max(processing_times)*1000:.3f} ms")
    print(f"  Block duration: {block_size/fs*1000:.3f} ms")
    print(f"  Real-time factor: {(block_size/fs)/np.mean(processing_times):.1f}x")
    
    print("✓ Real-time processing successful")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DSP FILTERING EXAMPLES")
    print("=" * 70)
    
    example1_individual_filters()
    example2_filter_cascade()
    example3_realtime_processing()
    
    print("\n" + "=" * 70)
    print("All DSP filtering examples complete!")
    print("Check data/outputs/ for results.")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()

