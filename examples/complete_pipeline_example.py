"""
Complete Pipeline Example

Demonstrates the full neural signal processing pipeline from signal generation
through ADC conversion, DMA buffering, filtering, spike detection, and visualization.

This example shows how all modules work together in a realistic embedded system simulation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from realtime_loop import RealtimeNeuralPipeline


def main():
    """Run complete pipeline example."""
    print("\n" + "=" * 70)
    print("COMPLETE NEURAL SIGNAL PROCESSING PIPELINE EXAMPLE")
    print("=" * 70)
    print()
    print("This example demonstrates a complete embedded neural recording system")
    print("with all components working together in real-time.")
    print()
    
    # Create pipeline with standard configuration
    pipeline = RealtimeNeuralPipeline(
        fs=20000.0,          # 20 kHz sampling
        block_size=512,      # 25.6 ms blocks
        adc_bits=12,         # 12-bit ADC
    )
    
    # Run simulation with different configurations
    
    print("\n" + "-" * 70)
    print("Example 1: Standard Recording (5 seconds, 30 Hz)")
    print("-" * 70)
    results1 = pipeline.run_simulation(
        duration=5.0,
        firing_rate=30.0,
        spike_type='biphasic',
        verbose=True
    )
    
    pipeline.export_results(results1, prefix='example1_standard')
    pipeline.visualize_results(results1, output_dir='data/outputs')
    
    print("\n" + "-" * 70)
    print("Example 2: High Activity (3 seconds, 80 Hz)")
    print("-" * 70)
    results2 = pipeline.run_simulation(
        duration=3.0,
        firing_rate=80.0,
        spike_type='triphasic',
        verbose=True
    )
    
    pipeline.export_results(results2, prefix='example2_high_activity')
    
    print("\n" + "-" * 70)
    print("Example 3: Low Activity (2 seconds, 10 Hz)")
    print("-" * 70)
    results3 = pipeline.run_simulation(
        duration=2.0,
        firing_rate=10.0,
        spike_type='simple',
        verbose=True
    )
    
    pipeline.export_results(results3, prefix='example3_low_activity')
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    print("Configuration          | Duration | Firing Rate | Detected | Precision | Recall")
    print("-" * 85)
    
    for name, results in [
        ("Standard (30 Hz)", results1),
        ("High Activity (80 Hz)", results2),
        ("Low Activity (10 Hz)", results3)
    ]:
        stats = results['statistics']
        print(f"{name:22} | {stats['duration_s']:6.1f}s | {stats['true_spikes']/stats['duration_s']:9.1f} Hz | "
              f"{stats['detected_spikes']:8} | {stats['precision']:8.1%} | {stats['recall']:6.1%}")
    
    print()
    print("=" * 70)
    print("All examples complete! Check data/outputs/ for results.")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()

