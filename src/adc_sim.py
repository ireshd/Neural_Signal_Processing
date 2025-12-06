"""
ADC Simulation Module

Simulates Analog-to-Digital Conversion (ADC) for neural signals with:
- Configurable sampling rates (10-20 kHz typical)
- Variable bit resolution (12-16 bits)
- Quantization effects
- Optional timing jitter
- Optional saturation/clipping behavior
- ADC noise simulation

This module mimics real embedded ADC hardware used in neural recording systems.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import warnings


class ADCSimulator:
    """
    Simulates an Analog-to-Digital Converter (ADC) for neural signals.
    
    Attributes:
        fs (float): Sampling frequency in Hz
        resolution (int): Bit resolution (e.g., 12, 14, 16 bits)
        vref_pos (float): Positive reference voltage
        vref_neg (float): Negative reference voltage
        enable_jitter (bool): Enable timing jitter
        enable_saturation (bool): Enable saturation/clipping
        adc_noise_std (float): ADC quantization noise standard deviation
    """
    
    def __init__(self,
                 fs: float = 20000.0,
                 resolution: int = 12,
                 vref_pos: float = 1.65,
                 vref_neg: float = -1.65,
                 enable_jitter: bool = False,
                 jitter_std: float = 1e-6,
                 enable_saturation: bool = True,
                 adc_noise_std: float = 0.0):
        """
        Initialize the ADC simulator.
        
        Args:
            fs: Sampling frequency in Hz (typical: 10-20 kHz for neural signals)
            resolution: Bit resolution (12, 14, or 16 bits typical)
            vref_pos: Positive reference voltage (V)
            vref_neg: Negative reference voltage (V)
            enable_jitter: Enable sampling time jitter
            jitter_std: Standard deviation of jitter (seconds)
            enable_saturation: Enable signal saturation at Vref limits
            adc_noise_std: ADC noise standard deviation (V)
        """
        self.fs = fs
        self.resolution = resolution
        self.vref_pos = vref_pos
        self.vref_neg = vref_neg
        self.enable_jitter = enable_jitter
        self.jitter_std = jitter_std
        self.enable_saturation = enable_saturation
        self.adc_noise_std = adc_noise_std
        
        # Calculate ADC parameters
        self.max_code = 2**resolution - 1
        self.voltage_range = vref_pos - vref_neg
        self.lsb = self.voltage_range / self.max_code  # Least Significant Bit
        self.offset = vref_neg
        
        # Statistics tracking
        self.stats = {
            'num_saturated_samples': 0,
            'num_samples_processed': 0,
            'saturation_percentage': 0.0
        }
    
    def analog_to_digital(self, analog_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert analog signal to digital codes.
        
        Args:
            analog_signal: Input analog signal (voltage)
            
        Returns:
            Tuple of (digital_codes, quantized_voltage)
        """
        signal = analog_signal.copy()
        
        # Apply saturation if enabled
        if self.enable_saturation:
            saturated_high = signal > self.vref_pos
            saturated_low = signal < self.vref_neg
            self.stats['num_saturated_samples'] = np.sum(saturated_high) + np.sum(saturated_low)
            
            signal = np.clip(signal, self.vref_neg, self.vref_pos)
        
        # Add ADC noise if specified
        if self.adc_noise_std > 0:
            signal += np.random.normal(0, self.adc_noise_std, len(signal))
        
        # Convert voltage to digital code
        # Code = (Voltage - Vref_neg) / LSB
        digital_codes = np.round((signal - self.offset) / self.lsb).astype(np.int32)
        
        # Ensure codes are within valid range
        digital_codes = np.clip(digital_codes, 0, self.max_code)
        
        # Convert back to quantized voltage (what the ADC actually represents)
        quantized_voltage = digital_codes * self.lsb + self.offset
        
        # Update statistics
        self.stats['num_samples_processed'] = len(signal)
        if self.stats['num_samples_processed'] > 0:
            self.stats['saturation_percentage'] = (
                self.stats['num_saturated_samples'] / self.stats['num_samples_processed'] * 100
            )
        
        return digital_codes, quantized_voltage
    
    def sample_signal(self,
                     time: np.ndarray,
                     analog_signal: np.ndarray,
                     target_fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample an analog signal at the ADC sampling rate with optional jitter.
        
        Args:
            time: Time array of input signal (seconds)
            analog_signal: Analog signal values
            target_fs: Target sampling frequency (uses self.fs if None)
            
        Returns:
            Tuple of (sample_times, sampled_analog, digital_codes, quantized_voltage)
        """
        if target_fs is None:
            target_fs = self.fs
        
        # Generate ideal sample times
        duration = time[-1] - time[0]
        num_samples = int(duration * target_fs)
        ideal_sample_times = np.linspace(time[0], time[-1], num_samples)
        
        # Add jitter if enabled
        if self.enable_jitter:
            jitter = np.random.normal(0, self.jitter_std, num_samples)
            actual_sample_times = ideal_sample_times + jitter
            # Ensure times stay within bounds and monotonic
            actual_sample_times = np.clip(actual_sample_times, time[0], time[-1])
            actual_sample_times = np.sort(actual_sample_times)
        else:
            actual_sample_times = ideal_sample_times
        
        # Interpolate analog signal at sample times
        sampled_analog = np.interp(actual_sample_times, time, analog_signal)
        
        # Convert to digital
        digital_codes, quantized_voltage = self.analog_to_digital(sampled_analog)
        
        return actual_sample_times, sampled_analog, digital_codes, quantized_voltage
    
    def quantize_only(self, analog_signal: np.ndarray) -> np.ndarray:
        """
        Apply only quantization to a signal (no sampling).
        
        Args:
            analog_signal: Input analog signal
            
        Returns:
            Quantized signal
        """
        _, quantized = self.analog_to_digital(analog_signal)
        return quantized
    
    def get_snr(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio due to quantization.
        
        Args:
            original: Original analog signal
            quantized: Quantized signal
            
        Returns:
            SNR in dB
        """
        # Quantization error
        error = original - quantized
        
        # Calculate SNR
        signal_power = np.mean(original**2)
        noise_power = np.mean(error**2)
        
        if noise_power == 0:
            return np.inf
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def get_enob(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Calculate Effective Number of Bits (ENOB).
        
        Args:
            original: Original analog signal
            quantized: Quantized signal
            
        Returns:
            ENOB value
        """
        snr_db = self.get_snr(original, quantized)
        # ENOB = (SNR - 1.76) / 6.02
        enob = (snr_db - 1.76) / 6.02
        return enob
    
    def get_statistics(self) -> Dict:
        """
        Get ADC statistics from last conversion.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.stats.copy()
        stats.update({
            'resolution_bits': self.resolution,
            'sampling_rate_hz': self.fs,
            'lsb_voltage': self.lsb,
            'voltage_range': self.voltage_range,
            'max_code': self.max_code,
            'vref_pos': self.vref_pos,
            'vref_neg': self.vref_neg
        })
        return stats
    
    def export_to_csv(self,
                     time: np.ndarray,
                     analog_signal: np.ndarray,
                     digital_codes: np.ndarray,
                     quantized_signal: np.ndarray,
                     output_dir: str = 'data/outputs',
                     prefix: str = 'adc_output') -> Dict[str, str]:
        """
        Export ADC data to CSV files.
        
        Args:
            time: Time array
            analog_signal: Original analog signal
            digital_codes: Digital codes from ADC
            quantized_signal: Quantized voltage values
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dictionary with paths to exported files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # 1. Export ADC data
        adc_data = np.column_stack((time, analog_signal, digital_codes, quantized_signal))
        data_path = os.path.join(output_dir, f'{prefix}_data.csv')
        np.savetxt(data_path, adc_data, delimiter=',',
                  header='time_s,analog_voltage,digital_code,quantized_voltage',
                  comments='')
        exported_files['adc_data'] = data_path
        
        # 2. Export quantization error
        error = analog_signal - quantized_signal
        error_data = np.column_stack((time, error))
        error_path = os.path.join(output_dir, f'{prefix}_quantization_error.csv')
        np.savetxt(error_path, error_data, delimiter=',',
                  header='time_s,error_voltage', comments='')
        exported_files['quantization_error'] = error_path
        
        # 3. Export statistics
        stats = self.get_statistics()
        snr = self.get_snr(analog_signal, quantized_signal)
        enob = self.get_enob(analog_signal, quantized_signal)
        
        stats_path = os.path.join(output_dir, f'{prefix}_statistics.csv')
        with open(stats_path, 'w') as f:
            f.write('parameter,value,unit\n')
            f.write(f'resolution,{self.resolution},bits\n')
            f.write(f'sampling_rate,{self.fs},Hz\n')
            f.write(f'vref_positive,{self.vref_pos},V\n')
            f.write(f'vref_negative,{self.vref_neg},V\n')
            f.write(f'voltage_range,{self.voltage_range},V\n')
            f.write(f'lsb_voltage,{self.lsb:.9f},V\n')
            f.write(f'max_code,{self.max_code},count\n')
            f.write(f'num_samples,{len(time)},count\n')
            f.write(f'saturated_samples,{stats["num_saturated_samples"]},count\n')
            f.write(f'saturation_percentage,{stats["saturation_percentage"]:.2f},%\n')
            f.write(f'snr,{snr:.2f},dB\n')
            f.write(f'enob,{enob:.2f},bits\n')
            f.write(f'jitter_enabled,{self.enable_jitter},boolean\n')
            f.write(f'jitter_std,{self.jitter_std:.9f},s\n')
            f.write(f'adc_noise_std,{self.adc_noise_std:.9f},V\n')
        exported_files['statistics'] = stats_path
        
        return exported_files


def demo():
    """
    Demonstration of the ADC simulator.
    """
    import matplotlib.pyplot as plt
    from signal_gen import NeuralSignalGenerator
    
    print("=" * 60)
    print("ADC Simulator Demo")
    print("=" * 60)
    
    # Generate a neural signal
    print("\n1. Generating neural signal...")
    gen = NeuralSignalGenerator(fs=100000, noise_amplitude=0.05, drift_amplitude=0.2)
    duration = 0.1  # 100 ms
    t_analog, signal_analog, spike_times, spike_indices = gen.generate_signal(
        duration=duration,
        firing_rate=50.0,
        spike_type='biphasic',
        seed=42
    )
    
    # Create ADC with different resolutions for comparison
    adc_configs = [
        {'resolution': 8, 'name': '8-bit'},
        {'resolution': 12, 'name': '12-bit'},
        {'resolution': 16, 'name': '16-bit'}
    ]
    
    fig, axes = plt.subplots(len(adc_configs) + 1, 2, figsize=(14, 12))
    
    # Plot original signal
    print("\n2. Converting with different ADC resolutions...")
    axes[0, 0].plot(t_analog * 1000, signal_analog, 'b-', linewidth=0.5, alpha=0.7)
    axes[0, 0].set_ylabel('Amplitude (V)')
    axes[0, 0].set_title('Original Analog Signal (100 kHz)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(signal_analog, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Amplitude (V)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Amplitude Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    results = []
    
    for idx, config in enumerate(adc_configs):
        # Create ADC
        adc = ADCSimulator(
            fs=20000,  # 20 kHz sampling
            resolution=config['resolution'],
            vref_pos=1.0,
            vref_neg=-1.0,
            enable_jitter=False,
            enable_saturation=True,
            adc_noise_std=0.001
        )
        
        # Sample and quantize the signal
        t_sampled, sampled_analog, digital_codes, quantized = adc.sample_signal(
            t_analog, signal_analog
        )
        
        # Get statistics
        snr = adc.get_snr(sampled_analog, quantized)
        enob = adc.get_enob(sampled_analog, quantized)
        stats = adc.get_statistics()
        
        print(f"\n{config['name']} ADC:")
        print(f"  LSB: {adc.lsb*1000:.6f} mV")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  ENOB: {enob:.2f} bits")
        print(f"  Saturation: {stats['saturation_percentage']:.2f}%")
        
        results.append({
            'time': t_sampled,
            'quantized': quantized,
            'error': sampled_analog - quantized,
            'snr': snr,
            'enob': enob,
            'lsb': adc.lsb
        })
        
        # Plot quantized signal
        row = idx + 1
        axes[row, 0].plot(t_sampled * 1000, quantized, 'r-', linewidth=0.5, alpha=0.7, label='Quantized')
        axes[row, 0].plot(t_sampled * 1000, sampled_analog, 'b.', markersize=1, alpha=0.3, label='Sampled')
        axes[row, 0].set_ylabel('Amplitude (V)')
        axes[row, 0].set_title(f'{config["name"]} ADC Output (SNR: {snr:.1f} dB, ENOB: {enob:.1f})')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)
        
        # Plot quantization error
        axes[row, 1].plot(t_sampled * 1000, results[idx]['error'] * 1000, 'g-', linewidth=0.5)
        axes[row, 1].axhline(y=adc.lsb*1000, color='r', linestyle='--', label=f'LSB ({adc.lsb*1000:.3f} mV)')
        axes[row, 1].axhline(y=-adc.lsb*1000, color='r', linestyle='--')
        axes[row, 1].set_ylabel('Error (mV)')
        axes[row, 1].set_title(f'Quantization Error (LSB: {adc.lsb*1000:.3f} mV)')
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 1].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    
    # Save output
    import os
    output_dir = 'data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'adc_simulation_demo.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n3. Plot saved to: {plot_path}")
    
    # Export CSV data for 12-bit ADC
    print("\n4. Exporting 12-bit ADC data to CSV...")
    adc_12bit = ADCSimulator(fs=20000, resolution=12, vref_pos=1.0, vref_neg=-1.0)
    t_sampled, sampled_analog, digital_codes, quantized = adc_12bit.sample_signal(
        t_analog, signal_analog
    )
    
    exported_files = adc_12bit.export_to_csv(
        t_sampled, sampled_analog, digital_codes, quantized,
        output_dir=output_dir,
        prefix='adc_12bit'
    )
    
    for key, path in exported_files.items():
        print(f"  ✓ {key}: {path}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    try:
        plt.show()
    except:
        pass


if __name__ == '__main__':
    demo()

