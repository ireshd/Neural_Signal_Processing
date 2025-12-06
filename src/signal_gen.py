"""
Neural Signal Generator

Generates synthetic neural signals with:
- Background noise (Gaussian white noise)
- Low-frequency drift (physiological baseline shifts)
- Action potential (spike) waveforms
- Configurable firing rates

This simulates extracellular recording signals typical of neural interfaces.
"""

import numpy as np
from typing import Optional, Tuple, List


class NeuralSignalGenerator:
    """
    Generates synthetic neural signals with noise, drift, and spike waveforms.
    
    Attributes:
        fs (float): Sampling frequency in Hz
        noise_amplitude (float): Standard deviation of background noise
        drift_amplitude (float): Amplitude of low-frequency drift
        drift_freq (float): Frequency of drift component in Hz
    """
    
    def __init__(self, 
                 fs: float = 20000.0,
                 noise_amplitude: float = 0.05,
                 drift_amplitude: float = 0.2,
                 drift_freq: float = 1.0):
        """
        Initialize the neural signal generator.
        
        Args:
            fs: Sampling frequency in Hz (default: 20 kHz)
            noise_amplitude: Standard deviation of Gaussian noise (default: 0.05)
            drift_amplitude: Amplitude of sinusoidal drift (default: 0.2)
            drift_freq: Frequency of drift component in Hz (default: 1 Hz)
        """
        self.fs = fs
        self.noise_amplitude = noise_amplitude
        self.drift_amplitude = drift_amplitude
        self.drift_freq = drift_freq
        
    def generate_noise(self, duration: float) -> np.ndarray:
        """
        Generate Gaussian white noise.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Noise signal array
        """
        N = int(duration * self.fs)
        return self.noise_amplitude * np.random.randn(N)
    
    def generate_drift(self, duration: float, phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate low-frequency drift component.
        
        Args:
            duration: Duration in seconds
            phase: Phase offset in radians
            
        Returns:
            Tuple of (time array, drift signal)
        """
        N = int(duration * self.fs)
        t = np.arange(N) / self.fs
        drift = self.drift_amplitude * np.sin(2 * np.pi * self.drift_freq * t + phase)
        return t, drift
    
    def generate_spike_waveform(self, spike_type: str = 'biphasic') -> np.ndarray:
        """
        Generate a single action potential waveform.
        
        Args:
            spike_type: Type of spike ('biphasic', 'triphasic', 'simple')
            
        Returns:
            Spike waveform array
        """
        # Spike duration: ~2 ms is typical for extracellular spikes
        spike_duration = 0.002  # seconds
        N_spike = int(spike_duration * self.fs)
        t_spike = np.linspace(0, spike_duration, N_spike)
        
        if spike_type == 'biphasic':
            # Classic biphasic action potential
            # Negative phase followed by positive phase
            tau1 = 0.0003  # decay time constant for negative phase
            tau2 = 0.0006  # decay time constant for positive phase
            
            # Negative peak at ~0.3 ms
            t_peak1 = 0.0003
            phase1 = -1.0 * np.exp(-(t_spike - t_peak1)**2 / (2 * tau1**2))
            
            # Positive peak at ~0.8 ms
            t_peak2 = 0.0008
            phase2 = 0.3 * np.exp(-(t_spike - t_peak2)**2 / (2 * tau2**2))
            
            spike = phase1 + phase2
            
        elif spike_type == 'triphasic':
            # Triphasic waveform: positive-negative-positive
            tau = 0.0002
            
            t_peak1 = 0.0002
            phase1 = 0.2 * np.exp(-(t_spike - t_peak1)**2 / (2 * tau**2))
            
            t_peak2 = 0.0005
            phase2 = -1.0 * np.exp(-(t_spike - t_peak2)**2 / (2 * (1.5*tau)**2))
            
            t_peak3 = 0.001
            phase3 = 0.15 * np.exp(-(t_spike - t_peak3)**2 / (2 * tau**2))
            
            spike = phase1 + phase2 + phase3
            
        else:  # 'simple'
            # Simple Gaussian-like spike
            t_peak = spike_duration / 2
            tau = 0.0003
            spike = -np.exp(-(t_spike - t_peak)**2 / (2 * tau**2))
        
        return spike
    
    def generate_spike_train(self, 
                            duration: float, 
                            firing_rate: float,
                            refractory_period: float = 0.002) -> np.ndarray:
        """
        Generate spike timing based on a Poisson process.
        
        Args:
            duration: Duration in seconds
            firing_rate: Average firing rate in Hz
            refractory_period: Minimum time between spikes in seconds
            
        Returns:
            Array of spike times in seconds
        """
        # Generate candidate spike times using Poisson process
        num_expected_spikes = int(duration * firing_rate * 1.5)  # Generate extra
        inter_spike_intervals = np.random.exponential(1.0 / firing_rate, num_expected_spikes)
        candidate_times = np.cumsum(inter_spike_intervals)
        
        # Enforce refractory period
        spike_times = []
        last_spike = -refractory_period
        
        for t in candidate_times:
            if t > duration:
                break
            if t - last_spike >= refractory_period:
                spike_times.append(t)
                last_spike = t
        
        return np.array(spike_times)
    
    def inject_spikes(self, 
                     base_signal: np.ndarray,
                     spike_times: np.ndarray,
                     spike_waveform: np.ndarray,
                     amplitude_variation: float = 0.1) -> np.ndarray:
        """
        Inject spike waveforms into the base signal.
        
        Args:
            base_signal: Base signal to inject spikes into
            spike_times: Array of spike times in seconds
            spike_waveform: Spike waveform template
            amplitude_variation: Random variation in spike amplitude (0-1)
            
        Returns:
            Signal with spikes injected
        """
        signal_with_spikes = base_signal.copy()
        spike_len = len(spike_waveform)
        
        for spike_time in spike_times:
            spike_idx = int(spike_time * self.fs)
            
            # Check bounds
            if spike_idx + spike_len >= len(base_signal):
                continue
            
            # Add random amplitude variation
            amplitude = 1.0 + amplitude_variation * (2 * np.random.rand() - 1)
            
            # Inject spike
            signal_with_spikes[spike_idx:spike_idx + spike_len] += amplitude * spike_waveform
        
        return signal_with_spikes
    
    def generate_signal(self,
                       duration: float,
                       firing_rate: float = 30.0,
                       spike_type: str = 'biphasic',
                       amplitude_variation: float = 0.1,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Generate a complete synthetic neural signal.
        
        Args:
            duration: Duration in seconds
            firing_rate: Average firing rate in Hz
            spike_type: Type of spike waveform ('biphasic', 'triphasic', 'simple')
            amplitude_variation: Random variation in spike amplitude (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (time array, complete signal, spike times, spike indices)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate base components
        noise = self.generate_noise(duration)
        t, drift = self.generate_drift(duration)
        base_signal = noise + drift
        
        # Generate and inject spikes
        spike_times = self.generate_spike_train(duration, firing_rate)
        spike_waveform = self.generate_spike_waveform(spike_type)
        
        # Create complete signal
        complete_signal = self.inject_spikes(base_signal, spike_times, spike_waveform, amplitude_variation)
        
        # Convert spike times to indices for easy plotting
        spike_indices = [int(t * self.fs) for t in spike_times if int(t * self.fs) < len(complete_signal)]
        
        return t, complete_signal, spike_times, spike_indices
    
    def generate_multi_unit_signal(self,
                                   duration: float,
                                   num_units: int = 3,
                                   firing_rates: Optional[List[float]] = None,
                                   spike_types: Optional[List[str]] = None,
                                   seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate a signal with multiple neuron units.
        
        Args:
            duration: Duration in seconds
            num_units: Number of neuron units to simulate
            firing_rates: List of firing rates for each unit (defaults to random 10-50 Hz)
            spike_types: List of spike types for each unit
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (time array, complete signal, list of spike times for each unit)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Default firing rates
        if firing_rates is None:
            firing_rates = [np.random.uniform(10, 50) for _ in range(num_units)]
        
        # Default spike types
        if spike_types is None:
            spike_types = ['biphasic'] * num_units
        
        # Generate base signal
        noise = self.generate_noise(duration)
        t, drift = self.generate_drift(duration)
        complete_signal = noise + drift
        
        all_spike_times = []
        
        # Add spikes from each unit
        for i in range(num_units):
            spike_times = self.generate_spike_train(duration, firing_rates[i])
            spike_waveform = self.generate_spike_waveform(spike_types[i])
            
            # Scale amplitude for different units
            amplitude_scale = 0.5 + 0.5 * (i / max(1, num_units - 1))
            scaled_waveform = spike_waveform * amplitude_scale
            
            complete_signal = self.inject_spikes(complete_signal, spike_times, scaled_waveform, amplitude_variation=0.15)
            all_spike_times.append(spike_times)
        
        return t, complete_signal, all_spike_times


def demo():
    """
    Demonstration of the neural signal generator.
    """
    import matplotlib.pyplot as plt
    
    # Create generator
    gen = NeuralSignalGenerator(fs=20000, noise_amplitude=0.05, drift_amplitude=0.2, drift_freq=1.0)
    
    # Generate signal
    duration = 1.0  # 1 second
    t, signal, spike_times, spike_indices = gen.generate_signal(
        duration=duration,
        firing_rate=30.0,
        spike_type='biphasic',
        seed=42
    )
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Full signal
    axes[0].plot(t, signal, 'b-', linewidth=0.5, label='Neural Signal')
    axes[0].plot(t[spike_indices], signal[spike_indices], 'r.', markersize=8, label='Spikes')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Synthetic Neural Signal with Spikes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Zoomed view
    zoom_duration = 0.1  # 100 ms
    zoom_samples = int(zoom_duration * gen.fs)
    axes[1].plot(t[:zoom_samples], signal[:zoom_samples], 'b-', linewidth=0.8)
    zoom_spikes = [idx for idx in spike_indices if idx < zoom_samples]
    if zoom_spikes:
        axes[1].plot(t[zoom_spikes], signal[zoom_spikes], 'r.', markersize=10)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Zoomed View (first {zoom_duration*1000:.0f} ms)')
    axes[1].grid(True, alpha=0.3)
    
    # Spike waveform template
    spike_waveform = gen.generate_spike_waveform('biphasic')
    t_spike = np.arange(len(spike_waveform)) / gen.fs * 1000  # Convert to ms
    axes[2].plot(t_spike, spike_waveform, 'k-', linewidth=2)
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Biphasic Spike Waveform Template')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to data/outputs directory (works with Docker volumes)
    import os
    output_dir = 'data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'neural_signal_demo.png')
    plt.savefig(output_path, dpi=150)
    
    print(f"Generated {len(spike_times)} spikes in {duration} seconds")
    print(f"Average firing rate: {len(spike_times)/duration:.1f} Hz")
    print(f"Plot saved to: {output_path}")
    
    # Try to show plot if interactive backend is available
    try:
        plt.show()
    except:
        pass


if __name__ == '__main__':
    demo()

