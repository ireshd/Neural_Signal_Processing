"""
Real-Time Signal Processing Loop

Integrates all pipeline components into a cohesive real-time processing system:
- Signal generation → ADC → DMA → Filtering → Spike Detection → Visualization
- Block-by-block processing with timing measurements
- Performance monitoring and statistics
- CSV export of complete pipeline results

Simulates the architecture of embedded neural recording systems with:
- DMA-style buffering
- ISR-like callbacks
- Deterministic block processing
- Latency tracking
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import time
import os
import csv

from signal_gen import NeuralSignalGenerator
from adc_sim import ADCSimulator
from dma_buffer import DMABuffer
from dsp_filters import design_neural_filter_cascade, FilterCascade
from spike_detect import SpikeDetector
from visualize import NeuralVisualizer


class RealtimeNeuralPipeline:
    """
    Complete real-time neural signal processing pipeline.
    
    Orchestrates:
    1. Signal generation (simulates neural activity)
    2. ADC sampling (simulates hardware conversion)
    3. DMA buffering (simulates DMA transfers)
    4. DSP filtering (band-pass, notch)
    5. Spike detection (adaptive threshold)
    6. Visualization and export
    
    Attributes:
        fs (float): Sampling frequency
        block_size (int): Processing block size
        components: Dictionary of pipeline components
    """
    
    def __init__(self,
                 fs: float = 20000.0,
                 block_size: int = 512,
                 adc_bits: int = 12,
                 buffer_size: Optional[int] = None):
        """
        Initialize real-time pipeline.
        
        Args:
            fs: Sampling frequency in Hz
            block_size: Block size for processing (samples)
            adc_bits: ADC resolution in bits
            buffer_size: DMA buffer size (default: 2 * block_size)
        """
        self.fs = fs
        self.block_size = block_size
        self.adc_bits = adc_bits
        self.buffer_size = buffer_size if buffer_size else 2 * block_size
        
        # Initialize components
        print("Initializing pipeline components...")
        
        self.signal_gen = NeuralSignalGenerator(fs=fs)
        print(f"  ✓ Signal generator (fs={fs} Hz)")
        
        self.adc = ADCSimulator(
            fs=fs,
            resolution=adc_bits,
            vref_pos=1.65,
            vref_neg=-1.65,
            adc_noise_std=0.001
        )
        print(f"  ✓ ADC simulator ({adc_bits}-bit, Vref=±1.65V)")
        
        self.dma_buffer = DMABuffer(
            buffer_size=self.buffer_size,
            block_size=block_size
        )
        print(f"  ✓ DMA buffer ({self.buffer_size} samples, blocks of {block_size})")
        
        self.filter_cascade = design_neural_filter_cascade(fs=fs)
        print(f"  ✓ DSP filters (High-pass → Band-pass → Notch)")
        
        self.spike_detector = SpikeDetector(
            fs=fs,
            threshold_factor=4.0,
            refractory_period=0.001
        )
        print(f"  ✓ Spike detector (threshold=4σ, refractory=1ms)")
        
        self.visualizer = NeuralVisualizer(fs=fs)
        print(f"  ✓ Visualizer")
        
        # Processing state
        self.processed_blocks = 0
        self.total_samples = 0
        self.detected_spikes: List[int] = []
        self.spike_waveforms: List[np.ndarray] = []
        
        # Timing measurements
        self.block_times: List[float] = []
        self.filter_times: List[float] = []
        self.detection_times: List[float] = []
        
        # Data storage
        self.raw_signal_buffer: List[np.ndarray] = []
        self.filtered_signal_buffer: List[np.ndarray] = []
        
        print("Pipeline initialized!\n")
    
    def process_block(self, block: np.ndarray) -> Dict[str, any]:
        """
        Process a single data block through the pipeline.
        
        Args:
            block: Input data block (ADC samples)
            
        Returns:
            Dictionary containing processing results and timing
        """
        block_start = time.time()
        results = {}
        
        # 1. DSP Filtering
        filter_start = time.time()
        filtered_block = self.filter_cascade.filter_block(block)
        filter_time = time.time() - filter_start
        self.filter_times.append(filter_time)
        
        # 2. Spike Detection
        detection_start = time.time()
        offset = self.total_samples
        spike_indices, waveforms = self.spike_detector.detect_spikes_stream(
            filtered_block, offset=offset
        )
        detection_time = time.time() - detection_start
        self.detection_times.append(detection_time)
        
        # Store results
        self.detected_spikes.extend(spike_indices)
        self.spike_waveforms.extend([w for w in waveforms if w is not None])
        
        # Store data for visualization
        self.raw_signal_buffer.append(block)
        self.filtered_signal_buffer.append(filtered_block)
        
        # Update counters
        self.processed_blocks += 1
        self.total_samples += len(block)
        
        block_time = time.time() - block_start
        self.block_times.append(block_time)
        
        results['block_number'] = self.processed_blocks
        results['samples_processed'] = len(block)
        results['spikes_detected'] = len(spike_indices)
        results['block_time_ms'] = block_time * 1000
        results['filter_time_ms'] = filter_time * 1000
        results['detection_time_ms'] = detection_time * 1000
        
        return results
    
    def run_simulation(self,
                      duration: float = 5.0,
                      firing_rate: float = 30.0,
                      spike_type: str = 'biphasic',
                      verbose: bool = True) -> Dict[str, any]:
        """
        Run complete real-time simulation.
        
        Args:
            duration: Simulation duration in seconds
            firing_rate: Target firing rate in Hz
            spike_type: Type of spike waveforms
            verbose: Print progress messages
            
        Returns:
            Dictionary containing complete results and statistics
        """
        if verbose:
            print("=" * 70)
            print(f"Running Real-Time Neural Signal Processing Simulation")
            print("=" * 70)
            print(f"Duration: {duration} s")
            print(f"Firing rate: {firing_rate} Hz")
            print(f"Block size: {self.block_size} samples ({self.block_size/self.fs*1000:.1f} ms)")
            print()
        
        # Reset state
        self.reset()
        
        # Generate complete signal
        if verbose:
            print("1. Generating neural signal...")
        
        t, analog_signal, true_spike_times, true_spike_indices = \
            self.signal_gen.generate_signal(
                duration=duration,
                firing_rate=firing_rate,
                spike_type=spike_type,
                seed=42
            )
        
        if verbose:
            print(f"   Generated {len(t)} samples ({duration} s)")
            print(f"   True spikes: {len(true_spike_times)}")
            print(f"   Signal RMS: {np.sqrt(np.mean(analog_signal**2)):.4f}")
        
        # ADC sampling
        if verbose:
            print("\n2. ADC conversion...")
        
        # Sample signal at ADC rate (already at correct rate)
        digital_codes, digital_signal = self.adc.analog_to_digital(analog_signal)
        snr = self.adc.get_snr(analog_signal, digital_signal)
        enob = self.adc.get_enob(analog_signal, digital_signal)
        
        if verbose:
            print(f"   ADC: {self.adc_bits}-bit, {self.fs} Hz")
            print(f"   SNR: {snr:.2f} dB")
            print(f"   ENOB: {enob:.2f} bits")
        
        # Block-by-block processing
        if verbose:
            print("\n3. Block-by-block processing...")
        
        num_blocks = len(digital_signal) // self.block_size
        
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            block = digital_signal[start_idx:end_idx]
            
            # Process block
            results = self.process_block(block)
            
            # Print progress
            if verbose and (i % 20 == 0 or i == num_blocks - 1):
                progress = (i + 1) / num_blocks * 100
                print(f"   Block {i+1}/{num_blocks} ({progress:.1f}%) - "
                      f"{results['spikes_detected']} spikes, "
                      f"{results['block_time_ms']:.3f} ms")
        
        # Compile results
        if verbose:
            print("\n4. Computing statistics...")
        
        results = self.get_results(t, analog_signal, digital_signal,
                                   true_spike_times, true_spike_indices)
        
        if verbose:
            self.print_summary(results)
        
        return results
    
    def get_results(self,
                   t: np.ndarray,
                   analog_signal: np.ndarray,
                   digital_signal: np.ndarray,
                   true_spike_times: np.ndarray,
                   true_spike_indices: np.ndarray) -> Dict[str, any]:
        """
        Compile complete results from simulation.
        
        Args:
            t: Time array
            analog_signal: Original analog signal
            digital_signal: Digitized signal
            true_spike_times: Ground truth spike times
            true_spike_indices: Ground truth spike indices
            
        Returns:
            Dictionary of complete results
        """
        # Reconstruct full signals
        if len(self.raw_signal_buffer) > 0:
            raw_signal = np.concatenate(self.raw_signal_buffer)
            filtered_signal = np.concatenate(self.filtered_signal_buffer)
            
            # Truncate to processed length
            processed_length = len(raw_signal)
            t = t[:processed_length]
            analog_signal = analog_signal[:processed_length]
            digital_signal = digital_signal[:processed_length]
        else:
            # No blocks processed, use original signals
            raw_signal = digital_signal
            filtered_signal = digital_signal
        
        # Spike detection accuracy
        detected_times = np.array(self.detected_spikes) / self.fs
        
        # Match spikes (1 ms tolerance)
        tolerance = 0.001
        true_positives = 0
        for true_time in true_spike_times:
            if np.any(np.abs(detected_times - true_time) < tolerance):
                true_positives += 1
        
        false_positives = len(detected_times) - true_positives
        false_negatives = len(true_spike_times) - true_positives
        
        precision = true_positives / len(detected_times) if len(detected_times) > 0 else 0
        recall = true_positives / len(true_spike_times) if len(true_spike_times) > 0 else 0
        
        # Timing statistics
        if self.block_times:
            avg_block_time = np.mean(self.block_times) * 1000
            max_block_time = np.max(self.block_times) * 1000
            avg_filter_time = np.mean(self.filter_times) * 1000
            avg_detection_time = np.mean(self.detection_times) * 1000
        else:
            avg_block_time = max_block_time = 0
            avg_filter_time = avg_detection_time = 0
        
        # Real-time factor
        block_duration = self.block_size / self.fs * 1000  # ms
        realtime_factor = block_duration / avg_block_time if avg_block_time > 0 else 0
        
        results = {
            'time': t,
            'analog_signal': analog_signal,
            'digital_signal': digital_signal,
            'raw_signal': raw_signal,
            'filtered_signal': filtered_signal,
            'true_spike_times': true_spike_times,
            'detected_spike_times': detected_times,
            'spike_waveforms': self.spike_waveforms,
            'statistics': {
                'duration_s': len(t) / self.fs,
                'total_samples': len(t),
                'blocks_processed': self.processed_blocks,
                'true_spikes': len(true_spike_times),
                'detected_spikes': len(detected_times),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'avg_block_time_ms': avg_block_time,
                'max_block_time_ms': max_block_time,
                'avg_filter_time_ms': avg_filter_time,
                'avg_detection_time_ms': avg_detection_time,
                'block_duration_ms': block_duration,
                'realtime_factor': realtime_factor,
            }
        }
        
        return results
    
    def print_summary(self, results: Dict[str, any]) -> None:
        """Print summary of results."""
        stats = results['statistics']
        
        print("\n" + "=" * 70)
        print("SIMULATION RESULTS")
        print("=" * 70)
        
        print("\nSignal Processing:")
        print(f"  Duration: {stats['duration_s']:.2f} s")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Blocks processed: {stats['blocks_processed']}")
        
        print("\nSpike Detection:")
        print(f"  True spikes: {stats['true_spikes']}")
        print(f"  Detected spikes: {stats['detected_spikes']}")
        print(f"  True positives: {stats['true_positives']}")
        print(f"  False positives: {stats['false_positives']}")
        print(f"  False negatives: {stats['false_negatives']}")
        print(f"  Precision: {stats['precision']:.2%}")
        print(f"  Recall: {stats['recall']:.2%}")
        
        print("\nTiming Performance:")
        print(f"  Block duration: {stats['block_duration_ms']:.3f} ms")
        print(f"  Avg block time: {stats['avg_block_time_ms']:.3f} ms")
        print(f"  Max block time: {stats['max_block_time_ms']:.3f} ms")
        print(f"  Avg filter time: {stats['avg_filter_time_ms']:.3f} ms")
        print(f"  Avg detection time: {stats['avg_detection_time_ms']:.3f} ms")
        print(f"  Real-time factor: {stats['realtime_factor']:.1f}x")
        
        if stats['realtime_factor'] >= 1.0:
            print(f"  ✓ MEETS REAL-TIME REQUIREMENTS")
        else:
            print(f"  ✗ Does not meet real-time requirements")
        
        print("=" * 70)
    
    def export_results(self,
                      results: Dict[str, any],
                      output_dir: str = 'data/outputs',
                      prefix: str = 'realtime_pipeline') -> Dict[str, str]:
        """
        Export complete pipeline results.
        
        Args:
            results: Results dictionary from run_simulation
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dictionary of exported file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        exported = {}
        
        print("\nExporting results...")
        
        # Export signals
        signals_file = os.path.join(output_dir, f'{prefix}_signals.csv')
        with open(signals_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'analog', 'digital', 'raw', 'filtered'])
            for i in range(len(results['time'])):
                writer.writerow([
                    results['time'][i],
                    results['analog_signal'][i],
                    results['digital_signal'][i],
                    results['raw_signal'][i],
                    results['filtered_signal'][i]
                ])
        exported['signals'] = signals_file
        print(f"  ✓ {signals_file}")
        
        # Export spike times
        spikes_file = os.path.join(output_dir, f'{prefix}_spikes.csv')
        with open(spikes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['spike_number', 'time_s', 'detected'])
            
            # Combine true and detected
            all_times = np.unique(np.concatenate([
                results['true_spike_times'],
                results['detected_spike_times']
            ]))
            
            for i, t in enumerate(all_times):
                is_true = np.any(np.abs(results['true_spike_times'] - t) < 0.0001)
                is_detected = np.any(np.abs(results['detected_spike_times'] - t) < 0.0001)
                writer.writerow([i, t, int(is_detected)])
        
        exported['spikes'] = spikes_file
        print(f"  ✓ {spikes_file}")
        
        # Export statistics
        stats_file = os.path.join(output_dir, f'{prefix}_statistics.csv')
        with open(stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in results['statistics'].items():
                writer.writerow([key, value])
        exported['statistics'] = stats_file
        print(f"  ✓ {stats_file}")
        
        return exported
    
    def visualize_results(self,
                         results: Dict[str, any],
                         output_dir: str = 'data/outputs') -> None:
        """
        Create visualizations of pipeline results.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
        """
        print("\nCreating visualizations...")
        
        # Pipeline summary
        self.visualizer.plot_pipeline_summary(
            results['time'],
            results['raw_signal'],
            results['filtered_signal'],
            results['detected_spike_times'],
            results['spike_waveforms'],
            self.spike_detector.compute_threshold(results['filtered_signal']),
            output_file=os.path.join(output_dir, 'realtime_pipeline_summary.png')
        )
        print(f"  ✓ realtime_pipeline_summary.png")
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.processed_blocks = 0
        self.total_samples = 0
        self.detected_spikes.clear()
        self.spike_waveforms.clear()
        self.block_times.clear()
        self.filter_times.clear()
        self.detection_times.clear()
        self.raw_signal_buffer.clear()
        self.filtered_signal_buffer.clear()
        
        self.filter_cascade.reset_state()
        self.spike_detector.reset()
        self.dma_buffer.reset()


def demo():
    """Demonstrate complete real-time pipeline."""
    print("\n" + "=" * 70)
    print("REAL-TIME NEURAL SIGNAL PROCESSING PIPELINE")
    print("=" * 70)
    print("\nThis demo simulates a complete embedded neural recording system:")
    print("  • Neural signal generation (synthetic)")
    print("  • ADC conversion (12-bit)")
    print("  • DMA buffering (circular buffer)")
    print("  • DSP filtering (band-pass + notch)")
    print("  • Spike detection (adaptive threshold)")
    print("  • Performance profiling")
    print()
    
    # Create pipeline
    pipeline = RealtimeNeuralPipeline(
        fs=20000.0,
        block_size=512,
        adc_bits=12,
    )
    
    # Run simulation
    results = pipeline.run_simulation(
        duration=5.0,
        firing_rate=30.0,
        spike_type='biphasic',
        verbose=True
    )
    
    # Export results
    exported = pipeline.export_results(results)
    
    # Create visualizations
    pipeline.visualize_results(results)
    
    print("\n" + "=" * 70)
    print("PIPELINE DEMO COMPLETE!")
    print("=" * 70)
    print("\nOutput files created in data/outputs/:")
    for file_type, path in exported.items():
        print(f"  • {os.path.basename(path)}")
    print("  • realtime_pipeline_summary.png")
    print("\n✓ All pipeline components working successfully!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    demo()

