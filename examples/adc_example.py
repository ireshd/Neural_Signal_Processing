"""
Example: ADC Simulation with Neural Signals

This example demonstrates how to use the ADC simulator to convert
analog neural signals to digital representations with various configurations.
"""

import sys
sys.path.insert(0, '..')

from src.signal_gen import NeuralSignalGenerator
from src.adc_sim import ADCSimulator
import numpy as np
import matplotlib.pyplot as plt


def basic_adc_example():
    """
    Basic example: Generate signal and digitize with ADC
    """
    print("=" * 60)
    print("Basic ADC Simulation Example")
    print("=" * 60)
    
    # Generate analog neural signal at high sampling rate
    print("\n1. Generating high-resolution analog signal...")
    gen = NeuralSignalGenerator(fs=100000)  # 100 kHz for "analog"
    duration = 0.5  # 500 ms
    
    t_analog, signal_analog, spike_times, spike_indices = gen.generate_signal(
        duration=duration,
        firing_rate=30.0,
        spike_type='biphasic',
        seed=42
    )
    
    # Create 12-bit ADC at 20 kHz
    print("\n2. Creating 12-bit ADC (20 kHz sampling)...")
    adc = ADCSimulator(
        fs=20000,           # 20 kHz sampling
        resolution=12,      # 12-bit resolution
        vref_pos=1.0,       # +1V reference
        vref_neg=-1.0,      # -1V reference
        enable_saturation=True
    )
    
    # Sample and digitize
    print("\n3. Sampling and digitizing signal...")
    t_digital, sampled_analog, digital_codes, quantized_voltage = adc.sample_signal(
        t_analog, signal_analog
    )
    
    # Calculate performance metrics
    snr = adc.get_snr(sampled_analog, quantized_voltage)
    enob = adc.get_enob(sampled_analog, quantized_voltage)
    
    print(f"\nADC Performance:")
    print(f"  Resolution: {adc.resolution} bits")
    print(f"  Sampling Rate: {adc.fs} Hz")
    print(f"  LSB: {adc.lsb * 1000:.6f} mV")
    print(f"  Voltage Range: {adc.voltage_range} V")
    print(f"  SNR: {snr:.2f} dB")
    print(f"  ENOB: {enob:.2f} bits")
    
    # Export data
    exported = adc.export_to_csv(
        t_digital, sampled_analog, digital_codes, quantized_voltage,
        output_dir='data/outputs',
        prefix='basic_adc'
    )
    
    print(f"\nExported {len(exported)} CSV files")


def compare_resolutions_example():
    """
    Compare different ADC resolutions
    """
    print("\n" + "=" * 60)
    print("ADC Resolution Comparison")
    print("=" * 60)
    
    # Generate signal
    gen = NeuralSignalGenerator(fs=100000)
    duration = 0.2
    t_analog, signal_analog, _, _ = gen.generate_signal(
        duration=duration, firing_rate=40.0, seed=123
    )
    
    resolutions = [8, 10, 12, 14, 16]
    results = []
    
    print("\nComparing ADC resolutions:")
    print("-" * 60)
    print(f"{'Resolution':<12} {'LSB (mV)':<12} {'SNR (dB)':<12} {'ENOB':<12}")
    print("-" * 60)
    
    for res in resolutions:
        adc = ADCSimulator(
            fs=20000,
            resolution=res,
            vref_pos=1.0,
            vref_neg=-1.0
        )
        
        t_dig, sampled, codes, quantized = adc.sample_signal(t_analog, signal_analog)
        snr = adc.get_snr(sampled, quantized)
        enob = adc.get_enob(sampled, quantized)
        
        print(f"{res:>4} bits    {adc.lsb*1000:>10.6f}  {snr:>10.2f}  {enob:>10.2f}")
        
        results.append({
            'resolution': res,
            'lsb': adc.lsb,
            'snr': snr,
            'enob': enob,
            'quantized': quantized
        })
    
    print("-" * 60)
    print("\n✓ Higher resolution = smaller LSB = better SNR")


def jitter_effect_example():
    """
    Demonstrate the effect of sampling jitter
    """
    print("\n" + "=" * 60)
    print("Sampling Jitter Effect")
    print("=" * 60)
    
    # Generate signal with high-frequency component
    gen = NeuralSignalGenerator(fs=100000, noise_amplitude=0.02)
    duration = 0.05  # 50 ms
    t_analog, signal_analog, _, _ = gen.generate_signal(
        duration=duration, firing_rate=100.0, seed=456
    )
    
    jitter_levels = [0, 1e-6, 5e-6, 10e-6]  # seconds
    
    print("\nComparing jitter levels:")
    print("-" * 60)
    print(f"{'Jitter (μs)':<15} {'SNR (dB)':<12} {'ENOB':<12}")
    print("-" * 60)
    
    for jitter_std in jitter_levels:
        adc = ADCSimulator(
            fs=20000,
            resolution=12,
            vref_pos=1.0,
            vref_neg=-1.0,
            enable_jitter=(jitter_std > 0),
            jitter_std=jitter_std
        )
        
        t_dig, sampled, codes, quantized = adc.sample_signal(t_analog, signal_analog)
        snr = adc.get_snr(sampled, quantized)
        enob = adc.get_enob(sampled, quantized)
        
        jitter_us = jitter_std * 1e6
        print(f"{jitter_us:>10.1f}     {snr:>10.2f}  {enob:>10.2f}")
    
    print("-" * 60)
    print("\n✓ Jitter reduces effective resolution (ENOB)")


def saturation_example():
    """
    Demonstrate signal saturation
    """
    print("\n" + "=" * 60)
    print("Signal Saturation Example")
    print("=" * 60)
    
    # Generate signal with large spikes
    gen = NeuralSignalGenerator(
        fs=100000,
        noise_amplitude=0.1,
        drift_amplitude=0.5  # Large drift
    )
    duration = 0.2
    t_analog, signal_analog, _, _ = gen.generate_signal(
        duration=duration, firing_rate=50.0, amplitude_variation=0.3, seed=789
    )
    
    # Scale signal to exceed ADC range
    signal_analog = signal_analog * 2.0  # Intentionally too large
    
    # ADC with saturation enabled
    print("\n1. With saturation enabled:")
    adc_sat = ADCSimulator(
        fs=20000,
        resolution=12,
        vref_pos=1.0,
        vref_neg=-1.0,
        enable_saturation=True
    )
    
    t_dig, sampled, codes, quantized = adc_sat.sample_signal(t_analog, signal_analog)
    stats = adc_sat.get_statistics()
    
    print(f"   Saturated samples: {stats['num_saturated_samples']}")
    print(f"   Saturation percentage: {stats['saturation_percentage']:.2f}%")
    print(f"   Signal range: [{np.min(signal_analog):.3f}, {np.max(signal_analog):.3f}] V")
    print(f"   ADC range: [{adc_sat.vref_neg:.3f}, {adc_sat.vref_pos:.3f}] V")
    
    print("\n✓ Signals exceeding Vref are clipped")


def adc_noise_example():
    """
    Demonstrate ADC noise
    """
    print("\n" + "=" * 60)
    print("ADC Noise Example")
    print("=" * 60)
    
    # Generate clean signal
    gen = NeuralSignalGenerator(fs=100000, noise_amplitude=0.01)
    duration = 0.1
    t_analog, signal_analog, _, _ = gen.generate_signal(
        duration=duration, firing_rate=20.0, seed=111
    )
    
    noise_levels = [0, 0.001, 0.005, 0.01]  # Volts
    
    print("\nComparing ADC noise levels:")
    print("-" * 60)
    print(f"{'Noise (mV)':<15} {'SNR (dB)':<12} {'ENOB':<12}")
    print("-" * 60)
    
    for noise_std in noise_levels:
        adc = ADCSimulator(
            fs=20000,
            resolution=12,
            vref_pos=1.0,
            vref_neg=-1.0,
            adc_noise_std=noise_std
        )
        
        t_dig, sampled, codes, quantized = adc.sample_signal(t_analog, signal_analog)
        snr = adc.get_snr(sampled, quantized)
        enob = adc.get_enob(sampled, quantized)
        
        print(f"{noise_std*1000:>10.1f}     {snr:>10.2f}  {enob:>10.2f}")
    
    print("-" * 60)
    print("\n✓ ADC noise degrades SNR and ENOB")


def realistic_neural_recording_example():
    """
    Simulate a realistic neural recording setup
    """
    print("\n" + "=" * 60)
    print("Realistic Neural Recording Setup")
    print("=" * 60)
    
    # Typical Intan RHD2000-series ADC specs:
    # - 16-bit resolution
    # - ±5 mV input range (for neural signals after amplification)
    # - 20 kHz sampling
    
    print("\nSimulating Intan RHD2000-like ADC:")
    print("  16-bit resolution")
    print("  20 kHz sampling rate")
    print("  ±5 mV input range")
    
    # Generate neural signal
    gen = NeuralSignalGenerator(
        fs=100000,
        noise_amplitude=0.01,  # 10 μV noise
        drift_amplitude=0.02   # 20 μV drift
    )
    
    duration = 1.0  # 1 second
    t_analog, signal_analog, spike_times, _ = gen.generate_signal(
        duration=duration,
        firing_rate=25.0,
        spike_type='biphasic',
        seed=999
    )
    
    # Scale to realistic amplitudes (neural signals after amplification)
    # Typical: 50-500 μV spikes → 0.05-0.5 mV after 1000x gain
    signal_analog = signal_analog * 0.005  # Scale to ±5 mV range
    
    # Create realistic ADC
    adc = ADCSimulator(
        fs=20000,
        resolution=16,
        vref_pos=0.005,      # +5 mV
        vref_neg=-0.005,     # -5 mV
        enable_jitter=True,
        jitter_std=0.5e-6,   # 0.5 μs jitter
        enable_saturation=True,
        adc_noise_std=0.0001 # 0.1 mV ADC noise
    )
    
    # Sample and digitize
    t_digital, sampled_analog, digital_codes, quantized_voltage = adc.sample_signal(
        t_analog, signal_analog
    )
    
    # Get performance metrics
    snr = adc.get_snr(sampled_analog, quantized_voltage)
    enob = adc.get_enob(sampled_analog, quantized_voltage)
    stats = adc.get_statistics()
    
    print(f"\nADC Performance:")
    print(f"  LSB: {adc.lsb * 1e6:.3f} μV")
    print(f"  SNR: {snr:.2f} dB")
    print(f"  ENOB: {enob:.2f} bits")
    print(f"  Samples: {len(t_digital):,}")
    print(f"  Spikes: {len(spike_times)}")
    print(f"  Saturation: {stats['saturation_percentage']:.2f}%")
    
    # Export data
    exported = adc.export_to_csv(
        t_digital, sampled_analog, digital_codes, quantized_voltage,
        output_dir='data/outputs',
        prefix='realistic_neural_recording'
    )
    
    print(f"\n✓ Exported realistic recording data:")
    for key, path in exported.items():
        print(f"  • {key}: {path}")


if __name__ == '__main__':
    # Run all examples
    basic_adc_example()
    compare_resolutions_example()
    jitter_effect_example()
    saturation_example()
    adc_noise_example()
    realistic_neural_recording_example()
    
    print("\n" + "=" * 60)
    print("All ADC examples completed!")
    print("Check 'data/outputs' for CSV files and plots")
    print("=" * 60)

