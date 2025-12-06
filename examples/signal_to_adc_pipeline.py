"""
Complete Signal-to-ADC Pipeline Example

This example demonstrates the full pipeline from signal generation
through ADC conversion, simulating a complete neural recording system.
"""

import sys
sys.path.insert(0, '..')

from src.signal_gen import NeuralSignalGenerator
from src.adc_sim import ADCSimulator
import numpy as np
import matplotlib.pyplot as plt


def complete_pipeline_example():
    """
    Full pipeline: Generate → Sample → Digitize → Export
    """
    print("=" * 70)
    print(" Complete Neural Recording Pipeline Simulation")
    print("=" * 70)
    
    # Step 1: Generate high-resolution analog signal
    print("\n[1/5] Generating analog neural signal...")
    print("      Settings: 100 kHz sampling (simulating continuous analog)")
    
    signal_gen = NeuralSignalGenerator(
        fs=100000,           # High rate for "analog" signal
        noise_amplitude=0.02,
        drift_amplitude=0.1,
        drift_freq=0.5
    )
    
    duration = 1.0  # 1 second recording
    t_analog, signal_analog, spike_times, spike_indices = signal_gen.generate_signal(
        duration=duration,
        firing_rate=35.0,
        spike_type='biphasic',
        seed=42
    )
    
    print(f"      ✓ Generated {len(t_analog):,} analog samples")
    print(f"      ✓ Detected {len(spike_times)} spikes")
    print(f"      ✓ Average firing rate: {len(spike_times)/duration:.1f} Hz")
    
    # Step 2: Configure ADC (simulating Intan-style neural recording)
    print("\n[2/5] Configuring ADC...")
    print("      Model: 16-bit, 20 kHz (similar to Intan RHD2000)")
    
    adc = ADCSimulator(
        fs=20000,            # 20 kHz sampling
        resolution=16,       # 16-bit resolution
        vref_pos=1.0,        # ±1V range
        vref_neg=-1.0,
        enable_jitter=True,
        jitter_std=0.5e-6,   # 0.5 μs jitter
        enable_saturation=True,
        adc_noise_std=0.0005 # 0.5 mV ADC noise
    )
    
    print(f"      ✓ Resolution: {adc.resolution} bits")
    print(f"      ✓ LSB: {adc.lsb * 1e6:.3f} μV")
    print(f"      ✓ Sampling rate: {adc.fs / 1000:.0f} kHz")
    print(f"      ✓ Voltage range: ±{adc.vref_pos} V")
    
    # Step 3: Sample and digitize
    print("\n[3/5] Sampling and digitizing signal...")
    
    t_digital, sampled_analog, digital_codes, quantized_voltage = adc.sample_signal(
        t_analog, signal_analog
    )
    
    print(f"      ✓ Digital samples: {len(t_digital):,}")
    print(f"      ✓ Downsampling ratio: {len(t_analog) / len(t_digital):.1f}x")
    
    # Step 4: Calculate performance metrics
    print("\n[4/5] Analyzing ADC performance...")
    
    snr = adc.get_snr(sampled_analog, quantized_voltage)
    enob = adc.get_enob(sampled_analog, quantized_voltage)
    stats = adc.get_statistics()
    
    print(f"      ✓ SNR: {snr:.2f} dB")
    print(f"      ✓ ENOB: {enob:.2f} bits")
    print(f"      ✓ Saturation: {stats['saturation_percentage']:.2f}%")
    
    quantization_error = sampled_analog - quantized_voltage
    print(f"      ✓ Max quantization error: {np.max(np.abs(quantization_error)) * 1e6:.2f} μV")
    print(f"      ✓ RMS quantization error: {np.sqrt(np.mean(quantization_error**2)) * 1e6:.2f} μV")
    
    # Step 5: Export all data
    print("\n[5/5] Exporting data to CSV...")
    
    # Export original signal
    signal_files = signal_gen.export_to_csv(
        t_analog, signal_analog, spike_times, spike_indices,
        output_dir='data/outputs',
        prefix='pipeline_analog'
    )
    
    # Export ADC output
    adc_files = adc.export_to_csv(
        t_digital, sampled_analog, digital_codes, quantized_voltage,
        output_dir='data/outputs',
        prefix='pipeline_digital'
    )
    
    print(f"      ✓ Signal files: {len(signal_files)}")
    for name, path in signal_files.items():
        print(f"         • {name}")
    
    print(f"      ✓ ADC files: {len(adc_files)}")
    for name, path in adc_files.items():
        print(f"         • {name}")
    
    # Visualization
    print("\n[Bonus] Creating visualization...")
    create_pipeline_visualization(
        t_analog, signal_analog, spike_indices,
        t_digital, quantized_voltage, digital_codes,
        adc
    )
    
    print("\n" + "=" * 70)
    print(" Pipeline Simulation Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  • Duration: {duration} second")
    print(f"  • Analog samples: {len(t_analog):,}")
    print(f"  • Digital samples: {len(t_digital):,}")
    print(f"  • Spikes detected: {len(spike_times)}")
    print(f"  • ADC performance: {snr:.1f} dB SNR, {enob:.1f} ENOB")
    print(f"  • Files exported: {len(signal_files) + len(adc_files)}")
    print(f"\n✓ All data saved to 'data/outputs/' directory")


def create_pipeline_visualization(t_analog, signal_analog, spike_indices,
                                  t_digital, quantized, digital_codes, adc):
    """
    Create comprehensive visualization of the pipeline
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Full analog signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_analog * 1000, signal_analog, 'b-', linewidth=0.3, alpha=0.7, label='Analog')
    ax1.plot(t_analog[spike_indices] * 1000, signal_analog[spike_indices], 
             'r.', markersize=4, label='Spikes')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (V)')
    ax1.set_title('Step 1: Generated Analog Neural Signal', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed comparison
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_start = 0.1
    zoom_end = 0.15
    zoom_mask_analog = (t_analog >= zoom_start) & (t_analog <= zoom_end)
    zoom_mask_digital = (t_digital >= zoom_start) & (t_digital <= zoom_end)
    
    ax2.plot(t_analog[zoom_mask_analog] * 1000, signal_analog[zoom_mask_analog], 
             'b-', linewidth=1, alpha=0.5, label='Analog')
    ax2.plot(t_digital[zoom_mask_digital] * 1000, quantized[zoom_mask_digital],
             'r.-', linewidth=1, markersize=3, label='Digitized')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (V)')
    ax2.set_title('Step 2: Analog vs Digital (Zoomed)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Digital codes
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_digital[zoom_mask_digital] * 1000, digital_codes[zoom_mask_digital],
             'g.-', linewidth=1, markersize=3)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('ADC Code')
    ax3.set_title('Digital Codes from ADC', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f'{adc.resolution}-bit\nMax: {adc.max_code}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Quantization error
    ax4 = fig.add_subplot(gs[2, 0])
    # Interpolate to compare
    analog_at_digital = np.interp(t_digital, t_analog, signal_analog)
    error = analog_at_digital - quantized
    ax4.plot(t_digital * 1000, error * 1e6, 'purple', linewidth=0.5)
    ax4.axhline(y=adc.lsb * 1e6, color='r', linestyle='--', alpha=0.5, label=f'LSB ({adc.lsb*1e6:.2f} μV)')
    ax4.axhline(y=-adc.lsb * 1e6, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Error (μV)')
    ax4.set_title('Step 3: Quantization Error', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Error histogram
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(error * 1e6, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax5.axvline(x=adc.lsb * 1e6, color='r', linestyle='--', label='LSB')
    ax5.axvline(x=-adc.lsb * 1e6, color='r', linestyle='--')
    ax5.set_xlabel('Error (μV)')
    ax5.set_ylabel('Count')
    ax5.set_title('Error Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    snr = adc.get_snr(analog_at_digital, quantized)
    enob = adc.get_enob(analog_at_digital, quantized)
    
    summary_text = f"""
    Pipeline Performance Summary
    {'─' * 60}
    
    Signal Generation:                        ADC Configuration:
      • Duration: {len(t_analog)/100000:.2f} s                           • Resolution: {adc.resolution} bits
      • Samples: {len(t_analog):,}                        • Sampling Rate: {adc.fs/1000:.0f} kHz
      • Spikes: {len(spike_indices)}                                • LSB: {adc.lsb*1e6:.3f} μV
                                                  • Voltage Range: ±{adc.vref_pos} V
    
    Performance Metrics:
      • SNR: {snr:.2f} dB
      • ENOB: {enob:.2f} bits (of {adc.resolution} nominal)
      • Max Quantization Error: {np.max(np.abs(error))*1e6:.2f} μV
      • RMS Quantization Error: {np.sqrt(np.mean(error**2))*1e6:.2f} μV
      • Compression: {len(t_analog)/len(t_digital):.1f}x (analog → digital samples)
    """
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Save
    import os
    output_dir = 'data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'pipeline_complete.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"      ✓ Visualization saved: {plot_path}")
    
    try:
        plt.show()
    except:
        pass


if __name__ == '__main__':
    complete_pipeline_example()

