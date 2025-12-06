"""
Example: Exporting Neural Signal Data to CSV

This example demonstrates how to generate neural signals and export
the data to CSV files for analysis in other tools (Excel, MATLAB, R, etc.)
"""

import sys
sys.path.insert(0, '..')

from src.signal_gen import NeuralSignalGenerator
import numpy as np


def basic_export_example():
    """
    Basic example: Generate signal and export to CSV
    """
    print("=" * 60)
    print("Basic CSV Export Example")
    print("=" * 60)
    
    # Create generator
    gen = NeuralSignalGenerator(
        fs=20000,           # 20 kHz sampling
        noise_amplitude=0.05,
        drift_amplitude=0.2,
        drift_freq=1.0
    )
    
    # Generate 2 seconds of signal
    duration = 2.0
    t, signal, spike_times, spike_indices = gen.generate_signal(
        duration=duration,
        firing_rate=25.0,  # 25 Hz firing rate
        spike_type='biphasic',
        seed=42
    )
    
    # Export to CSV
    exported_files = gen.export_to_csv(
        t, signal, spike_times, spike_indices,
        output_dir='data/outputs',
        prefix='example_basic'
    )
    
    print(f"\nExported {len(exported_files)} CSV files:")
    for key, path in exported_files.items():
        print(f"  • {key}: {path}")
    
    # Export waveform template
    waveform_path = gen.export_waveform_to_csv(
        spike_type='biphasic',
        output_dir='data/outputs',
        filename='example_waveform_biphasic.csv'
    )
    print(f"  • waveform: {waveform_path}")
    
    print(f"\n✓ Generated {len(spike_times)} spikes ({len(spike_times)/duration:.1f} Hz)")


def multi_unit_export_example():
    """
    Advanced example: Multi-unit recording with separate exports
    """
    print("\n" + "=" * 60)
    print("Multi-Unit Recording CSV Export Example")
    print("=" * 60)
    
    # Create generator
    gen = NeuralSignalGenerator(fs=20000)
    
    # Generate multi-unit signal
    duration = 1.0
    num_units = 3
    firing_rates = [15.0, 30.0, 45.0]  # Hz
    
    t, signal, all_spike_times = gen.generate_multi_unit_signal(
        duration=duration,
        num_units=num_units,
        firing_rates=firing_rates,
        spike_types=['biphasic', 'triphasic', 'simple'],
        seed=123
    )
    
    # Combine all spike times for export
    combined_spike_times = np.concatenate(all_spike_times)
    combined_spike_times.sort()
    spike_indices = [int(t * gen.fs) for t in combined_spike_times if int(t * gen.fs) < len(signal)]
    
    # Export combined signal
    exported_files = gen.export_to_csv(
        t, signal, combined_spike_times, spike_indices,
        output_dir='data/outputs',
        prefix='example_multiunit'
    )
    
    print(f"\nExported multi-unit recording:")
    for key, path in exported_files.items():
        print(f"  • {key}: {path}")
    
    # Export individual unit spike times
    import os
    for i, unit_spike_times in enumerate(all_spike_times):
        unit_path = os.path.join('data/outputs', f'example_unit{i+1}_spikes.csv')
        unit_indices = [int(t * gen.fs) for t in unit_spike_times]
        if len(unit_spike_times) > 0:
            unit_data = np.column_stack((unit_spike_times, unit_indices))
            np.savetxt(unit_path, unit_data, delimiter=',',
                      header='spike_time_s,spike_index', comments='', fmt=['%.6f', '%d'])
            print(f"  • unit {i+1}: {unit_path} ({len(unit_spike_times)} spikes, {firing_rates[i]} Hz)")
    
    print(f"\n✓ Generated {num_units} units with {len(combined_spike_times)} total spikes")


def compare_spike_types_example():
    """
    Example: Generate and export different spike waveform types
    """
    print("\n" + "=" * 60)
    print("Spike Waveform Comparison CSV Export")
    print("=" * 60)
    
    gen = NeuralSignalGenerator(fs=20000)
    spike_types = ['biphasic', 'triphasic', 'simple']
    
    print("\nExporting spike waveform templates:")
    for spike_type in spike_types:
        path = gen.export_waveform_to_csv(
            spike_type=spike_type,
            output_dir='data/outputs',
            filename=f'waveform_{spike_type}.csv'
        )
        print(f"  • {spike_type}: {path}")
    
    print("\n✓ All waveform types exported")


def batch_export_example():
    """
    Example: Generate multiple trials and export each
    """
    print("\n" + "=" * 60)
    print("Batch Export Example (Multiple Trials)")
    print("=" * 60)
    
    gen = NeuralSignalGenerator(fs=20000)
    
    num_trials = 5
    duration = 0.5  # 500 ms per trial
    
    print(f"\nGenerating and exporting {num_trials} trials:")
    for trial in range(num_trials):
        t, signal, spike_times, spike_indices = gen.generate_signal(
            duration=duration,
            firing_rate=30.0,
            spike_type='biphasic',
            seed=trial  # Different seed for each trial
        )
        
        exported_files = gen.export_to_csv(
            t, signal, spike_times, spike_indices,
            output_dir='data/outputs',
            prefix=f'trial_{trial+1:02d}'
        )
        
        print(f"  • Trial {trial+1}: {len(spike_times)} spikes, {len(exported_files)} files")
    
    print(f"\n✓ Exported {num_trials} trials")


if __name__ == '__main__':
    # Run all examples
    basic_export_example()
    multi_unit_export_example()
    compare_spike_types_example()
    batch_export_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check the 'data/outputs' directory for CSV files")
    print("=" * 60)

