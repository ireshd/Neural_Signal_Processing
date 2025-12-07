"""
Visualization Tools for Neural Signal Processing

Provides comprehensive visualization for neural signal processing pipeline:
- Real-time waveform plotting
- Spike raster plots
- Firing rate histograms
- FFT/Power Spectral Density
- Spectrograms
- Filter response visualization
- Performance metrics dashboard

Supports both matplotlib (static) and real-time plotting backends.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker/server environments
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import os


class NeuralVisualizer:
    """
    Comprehensive visualization tool for neural signal processing.
    
    Provides methods for creating publication-quality plots of:
    - Raw and filtered signals
    - Spike detection results
    - Spectral analysis
    - Performance metrics
    """
    
    def __init__(self, fs: float = 20000.0, figsize: Tuple[float, float] = (14, 10)):
        """
        Initialize visualizer.
        
        Args:
            fs: Sampling frequency in Hz
            figsize: Default figure size (width, height) in inches
        """
        self.fs = fs
        self.figsize = figsize
        
    def plot_signal_comparison(self,
                              t: np.ndarray,
                              raw_signal: np.ndarray,
                              filtered_signal: Optional[np.ndarray] = None,
                              spike_times: Optional[np.ndarray] = None,
                              output_file: Optional[str] = None,
                              title: str = "Neural Signal") -> None:
        """
        Plot raw and filtered signal comparison.
        
        Args:
            t: Time array in seconds
            raw_signal: Raw signal data
            filtered_signal: Filtered signal (optional)
            spike_times: Detected spike times in seconds (optional)
            output_file: Path to save figure (if None, shows interactively)
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Raw signal
        axes[0].plot(t, raw_signal, color='blue', alpha=0.7, linewidth=0.5)
        if spike_times is not None:
            axes[0].scatter(spike_times, np.zeros_like(spike_times),
                          c='red', marker='v', s=50, label='Spikes', zorder=5)
        axes[0].set_ylabel('Amplitude (Raw)')
        axes[0].set_title(f'{title} - Raw Signal')
        axes[0].grid(True, alpha=0.3)
        if spike_times is not None:
            axes[0].legend()
        
        # Filtered signal
        if filtered_signal is not None:
            axes[1].plot(t, filtered_signal, color='green', alpha=0.7, linewidth=0.5)
            if spike_times is not None:
                axes[1].scatter(spike_times, np.zeros_like(spike_times),
                              c='red', marker='v', s=50, label='Spikes', zorder=5)
            axes[1].set_ylabel('Amplitude (Filtered)')
            axes[1].set_title(f'{title} - Filtered Signal')
            axes[1].grid(True, alpha=0.3)
            if spike_times is not None:
                axes[1].legend()
        
        axes[-1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spike_detection(self,
                           t: np.ndarray,
                           signal: np.ndarray,
                           spike_times: np.ndarray,
                           threshold: float,
                           waveforms: Optional[List[np.ndarray]] = None,
                           output_file: Optional[str] = None,
                           title: str = "Spike Detection") -> None:
        """
        Plot spike detection results.
        
        Args:
            t: Time array in seconds
            signal: Signal data
            spike_times: Detected spike times in seconds
            threshold: Detection threshold
            waveforms: Extracted spike waveforms (optional)
            output_file: Path to save figure
            title: Plot title
        """
        if waveforms is not None and len(waveforms) > 0:
            fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 0.7))
            axes = [axes[0], axes[1], None]
        
        # Full signal with threshold
        axes[0].plot(t, signal, linewidth=0.5)
        axes[0].axhline(threshold, color='red', linestyle='--', 
                       label=f'Threshold ({threshold:.3f})', alpha=0.7)
        axes[0].axhline(-threshold, color='red', linestyle='--', alpha=0.7)
        axes[0].scatter(spike_times, np.zeros_like(spike_times),
                       c='red', marker='v', s=50, label=f'{len(spike_times)} spikes', zorder=5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{title} - Signal with Detected Spikes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Raster plot
        if len(spike_times) > 0:
            axes[1].eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='black')
            axes[1].set_ylabel('Spikes')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_title('Spike Raster Plot')
            axes[1].set_ylim([0, 1])
            axes[1].set_yticks([])
            axes[1].grid(True, alpha=0.3, axis='x')
        
        # Spike waveforms
        if waveforms is not None and len(waveforms) > 0 and axes[2] is not None:
            waveform_time = np.arange(len(waveforms[0])) / self.fs * 1000
            
            # Plot individual waveforms
            for waveform in waveforms[:min(100, len(waveforms))]:
                axes[2].plot(waveform_time, waveform, alpha=0.2, color='blue')
            
            # Mean waveform
            mean_waveform = np.mean(waveforms, axis=0)
            std_waveform = np.std(waveforms, axis=0)
            axes[2].plot(waveform_time, mean_waveform, color='red', 
                        linewidth=2, label='Mean')
            axes[2].fill_between(waveform_time,
                                mean_waveform - std_waveform,
                                mean_waveform + std_waveform,
                                alpha=0.3, color='red', label='±1 SD')
            axes[2].set_xlabel('Time (ms)')
            axes[2].set_ylabel('Amplitude')
            axes[2].set_title(f'Spike Waveforms (n={len(waveforms)})')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spectrogram(self,
                        signal: np.ndarray,
                        nperseg: int = 512,
                        output_file: Optional[str] = None,
                        title: str = "Spectrogram") -> None:
        """
        Plot spectrogram of signal.
        
        Args:
            signal: Input signal
            nperseg: Length of each segment for STFT
            output_file: Path to save figure
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # Time domain signal
        t = np.arange(len(signal)) / self.fs
        axes[0].plot(t, signal, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{title} - Time Domain')
        axes[0].grid(True, alpha=0.3)
        
        # Spectrogram
        f, t_spec, Sxx = scipy_signal.spectrogram(signal, fs=self.fs, nperseg=nperseg)
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        im = axes[1].pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='viridis')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title(f'{title} - Spectrogram')
        axes[1].set_ylim([0, min(5000, self.fs/2)])
        
        plt.colorbar(im, ax=axes[1], label='Power (dB)')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_psd(self,
                signals: Dict[str, np.ndarray],
                nperseg: int = 1024,
                output_file: Optional[str] = None,
                title: str = "Power Spectral Density") -> None:
        """
        Plot power spectral density of one or more signals.
        
        Args:
            signals: Dictionary of {label: signal_data}
            nperseg: Length of each segment for Welch's method
            output_file: Path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.5))
        
        for label, sig in signals.items():
            f, Pxx = scipy_signal.welch(sig, fs=self.fs, nperseg=nperseg)
            ax.semilogy(f, Pxx, label=label, alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (V²/Hz)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(5000, self.fs/2)])
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_filter_response(self,
                           filters: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           output_file: Optional[str] = None,
                           title: str = "Filter Frequency Response") -> None:
        """
        Plot frequency response of filters.
        
        Args:
            filters: Dictionary of {label: (b, a)} coefficient pairs
            output_file: Path to save figure
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        for label, (b, a) in filters.items():
            w, h = scipy_signal.freqz(b, a, worN=2048)
            freq = w * self.fs / (2 * np.pi)
            
            # Magnitude
            magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
            axes[0].plot(freq, magnitude_db, label=label, linewidth=2)
            
            # Phase
            phase = np.unwrap(np.angle(h))
            axes[1].plot(freq, phase, label=label, linewidth=2)
        
        axes[0].set_ylabel('Magnitude (dB)')
        axes[0].set_title(f'{title} - Magnitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Phase (radians)')
        axes[1].set_title(f'{title} - Phase')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_firing_rate(self,
                        spike_times: np.ndarray,
                        duration: float,
                        bin_size: float = 0.05,
                        output_file: Optional[str] = None,
                        title: str = "Firing Rate") -> None:
        """
        Plot firing rate over time.
        
        Args:
            spike_times: Array of spike times in seconds
            duration: Total duration in seconds
            bin_size: Bin size for histogram in seconds
            output_file: Path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.4))
        
        # Create histogram
        bins = np.arange(0, duration + bin_size, bin_size)
        counts, edges = np.histogram(spike_times, bins=bins)
        
        # Convert to firing rate (Hz)
        firing_rate = counts / bin_size
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        ax.bar(bin_centers, firing_rate, width=bin_size*0.8, 
               alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(f'{title} (bin size: {bin_size*1000:.0f} ms)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_rate = len(spike_times) / duration
        ax.axhline(mean_rate, color='red', linestyle='--', 
                  label=f'Mean: {mean_rate:.1f} Hz', linewidth=2)
        ax.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pipeline_summary(self,
                            t: np.ndarray,
                            raw_signal: np.ndarray,
                            filtered_signal: np.ndarray,
                            spike_times: np.ndarray,
                            waveforms: List[np.ndarray],
                            threshold: float,
                            output_file: str = 'data/outputs/pipeline_summary.png') -> None:
        """
        Create comprehensive summary plot of entire pipeline.
        
        Args:
            t: Time array in seconds
            raw_signal: Raw signal
            filtered_signal: Filtered signal
            spike_times: Detected spike times
            waveforms: Extracted spike waveforms
            threshold: Detection threshold
            output_file: Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. Raw signal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, raw_signal, linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Raw Neural Signal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Filtered signal with spikes
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(t, filtered_signal, linewidth=0.5)
        ax2.axhline(threshold, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(-threshold, color='red', linestyle='--', alpha=0.5)
        if len(spike_times) > 0:
            ax2.scatter(spike_times, np.zeros_like(spike_times),
                       c='red', marker='v', s=30, zorder=5)
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Filtered Signal with Detected Spikes (n={len(spike_times)})')
        ax2.grid(True, alpha=0.3)
        
        # 3. Spike raster
        ax3 = fig.add_subplot(gs[2, :])
        if len(spike_times) > 0:
            ax3.eventplot(spike_times, lineoffsets=0.5, linelengths=0.8, colors='black')
        ax3.set_ylabel('Spikes')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Spike Raster Plot')
        ax3.set_ylim([0, 1])
        ax3.set_yticks([])
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Spike waveforms
        ax4 = fig.add_subplot(gs[3, 0])
        if waveforms and len(waveforms) > 0:
            waveform_time = np.arange(len(waveforms[0])) / self.fs * 1000
            for waveform in waveforms[:min(50, len(waveforms))]:
                ax4.plot(waveform_time, waveform, alpha=0.2, color='blue')
            mean_waveform = np.mean(waveforms, axis=0)
            ax4.plot(waveform_time, mean_waveform, color='red', linewidth=2, label='Mean')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Amplitude')
            ax4.set_title(f'Spike Waveforms (n={len(waveforms)})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Firing rate histogram
        ax5 = fig.add_subplot(gs[3, 1])
        if len(spike_times) > 0:
            duration = t[-1]
            bin_size = 0.1  # 100 ms bins
            bins = np.arange(0, duration + bin_size, bin_size)
            counts, edges = np.histogram(spike_times, bins=bins)
            firing_rate = counts / bin_size
            bin_centers = (edges[:-1] + edges[1:]) / 2
            ax5.bar(bin_centers, firing_rate, width=bin_size*0.8, alpha=0.7)
            mean_rate = len(spike_times) / duration
            ax5.axhline(mean_rate, color='red', linestyle='--', 
                       label=f'Mean: {mean_rate:.1f} Hz', linewidth=2)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Firing Rate (Hz)')
            ax5.set_title('Firing Rate Over Time')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()


def demo():
    """Demonstrate visualization functionality."""
    print("=" * 70)
    print("Visualization Demo")
    print("=" * 70)
    
    # Generate test data
    from signal_gen import NeuralSignalGenerator
    from dsp_filters import design_neural_filter_cascade
    from spike_detect import SpikeDetector
    
    print("\n1. Generating Test Data")
    fs = 20000.0
    duration = 2.0
    
    gen = NeuralSignalGenerator(fs=fs)
    t, raw_signal, true_spike_times, _ = gen.generate_signal(
        duration=duration,
        firing_rate=30.0,
        spike_type='biphasic',
        seed=42
    )
    
    print(f"   Duration: {duration} s")
    print(f"   Samples: {len(t)}")
    print(f"   True spikes: {len(true_spike_times)}")
    
    # Filter signal
    print("\n2. Filtering Signal")
    filter_cascade = design_neural_filter_cascade(fs=fs)
    filtered_signal = filter_cascade.filter_block(raw_signal)
    
    # Detect spikes
    print("\n3. Detecting Spikes")
    detector = SpikeDetector(fs=fs)
    spike_indices, waveforms = detector.detect_spikes(filtered_signal)
    spike_times = spike_indices / fs
    threshold = detector.compute_threshold(filtered_signal)
    
    valid_waveforms = [w for w in waveforms if w is not None]
    
    print(f"   Detected: {len(spike_times)} spikes")
    print(f"   Extracted: {len(valid_waveforms)} waveforms")
    
    # Create visualizer
    print("\n4. Creating Visualizations")
    viz = NeuralVisualizer(fs=fs)
    
    # Signal comparison
    print("   - Signal comparison plot")
    viz.plot_signal_comparison(
        t, raw_signal, filtered_signal, spike_times,
        output_file='data/outputs/viz_signal_comparison.png',
        title='Neural Signal Processing'
    )
    
    # Spike detection
    print("   - Spike detection plot")
    viz.plot_spike_detection(
        t, filtered_signal, spike_times, threshold, valid_waveforms,
        output_file='data/outputs/viz_spike_detection.png',
        title='Spike Detection Results'
    )
    
    # Spectrogram
    print("   - Spectrogram")
    viz.plot_spectrogram(
        raw_signal,
        output_file='data/outputs/viz_spectrogram.png',
        title='Neural Signal Spectrogram'
    )
    
    # PSD
    print("   - Power spectral density")
    viz.plot_psd(
        {'Raw': raw_signal, 'Filtered': filtered_signal},
        output_file='data/outputs/viz_psd.png',
        title='Power Spectral Density Comparison'
    )
    
    # Firing rate
    print("   - Firing rate histogram")
    viz.plot_firing_rate(
        spike_times, duration, bin_size=0.05,
        output_file='data/outputs/viz_firing_rate.png',
        title='Firing Rate Over Time'
    )
    
    # Pipeline summary
    print("   - Pipeline summary")
    viz.plot_pipeline_summary(
        t, raw_signal, filtered_signal, spike_times, valid_waveforms, threshold,
        output_file='data/outputs/viz_pipeline_summary.png'
    )
    
    print("\n5. Visualization Files Created:")
    output_files = [
        'data/outputs/viz_signal_comparison.png',
        'data/outputs/viz_spike_detection.png',
        'data/outputs/viz_spectrogram.png',
        'data/outputs/viz_psd.png',
        'data/outputs/viz_firing_rate.png',
        'data/outputs/viz_pipeline_summary.png'
    ]
    
    for f in output_files:
        if os.path.exists(f):
            print(f"   ✓ {f}")
        else:
            print(f"   ✗ {f} (not created)")
    
    print("\n" + "=" * 70)
    print("Demo complete! Check data/outputs/ for visualization files.")
    print("=" * 70)


if __name__ == '__main__':
    demo()

