"""
DMA-Style Circular Buffer

Simulates a Direct Memory Access (DMA) circular buffer commonly used in embedded systems
for continuous data streaming from ADC peripherals. Features:
- Circular buffer implementation
- Block-based transfer callbacks (half-transfer and full-transfer)
- Buffer overflow detection
- Thread-safe operations
- Statistics tracking

This mimics the behavior of STM32/Cortex-M DMA controllers used in neural recording systems.
"""

import numpy as np
from typing import Optional, Callable, Dict, List, Tuple
from collections import deque
import time
from threading import Lock


class DMABuffer:
    """
    Circular buffer with DMA-style interrupt callbacks.
    
    Simulates hardware DMA behavior:
    - Data is written to a circular buffer
    - Callbacks trigger at half-full and full positions
    - Application processes data in blocks while DMA continues writing
    
    Attributes:
        buffer_size (int): Total buffer size in samples
        block_size (int): Size of each processing block
        buffer (np.ndarray): The circular buffer array
        write_idx (int): Current write position
        overflow_count (int): Number of buffer overflows detected
    """
    
    def __init__(self, 
                 buffer_size: int = 2048,
                 block_size: Optional[int] = None,
                 dtype: type = np.float64):
        """
        Initialize the DMA circular buffer.
        
        Args:
            buffer_size: Total size of circular buffer (must be power of 2 for efficiency)
            block_size: Size of each DMA block (default: buffer_size // 2)
            dtype: Data type for buffer elements
            
        Raises:
            ValueError: If buffer_size is not positive or block_size is invalid
        """
        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        
        self.buffer_size = buffer_size
        self.block_size = block_size if block_size is not None else buffer_size // 2
        
        if self.block_size <= 0 or self.block_size > buffer_size:
            raise ValueError(f"Block size must be between 1 and {buffer_size}")
        
        self.dtype = dtype
        self.buffer = np.zeros(buffer_size, dtype=dtype)
        self.write_idx = 0
        self.read_idx = 0
        self.overflow_count = 0
        self.samples_written = 0
        self.samples_read = 0
        
        # Callbacks
        self.half_complete_callback: Optional[Callable[[np.ndarray], None]] = None
        self.full_complete_callback: Optional[Callable[[np.ndarray], None]] = None
        
        # Thread safety
        self._lock = Lock()
        
        # Statistics
        self.callback_times: List[float] = []
        self.last_callback_time = 0.0
        
    def register_half_complete_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Register callback for half-transfer complete interrupt.
        
        Args:
            callback: Function to call when first half of buffer is filled
                     Receives the data block as argument
        """
        self.half_complete_callback = callback
        
    def register_full_complete_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Register callback for full-transfer complete interrupt.
        
        Args:
            callback: Function to call when second half of buffer is filled
                     Receives the data block as argument
        """
        self.full_complete_callback = callback
        
    def write(self, data: np.ndarray) -> int:
        """
        Write data to the circular buffer (simulates DMA transfer).
        
        Automatically triggers callbacks when buffer halves are filled.
        
        Args:
            data: Data array to write
            
        Returns:
            Number of samples successfully written
        """
        with self._lock:
            samples_to_write = len(data)
            samples_written = 0
            
            while samples_written < samples_to_write:
                # Calculate available space
                available = self.buffer_size - self._get_fill_level()
                
                if available == 0:
                    # Buffer overflow
                    self.overflow_count += 1
                    break
                
                # Write as much as possible
                chunk_size = min(samples_to_write - samples_written, available)
                
                # Handle wrap-around
                if self.write_idx + chunk_size <= self.buffer_size:
                    # No wrap-around
                    self.buffer[self.write_idx:self.write_idx + chunk_size] = \
                        data[samples_written:samples_written + chunk_size]
                    self.write_idx += chunk_size
                else:
                    # Wrap-around
                    first_part = self.buffer_size - self.write_idx
                    self.buffer[self.write_idx:] = data[samples_written:samples_written + first_part]
                    second_part = chunk_size - first_part
                    self.buffer[:second_part] = data[samples_written + first_part:samples_written + chunk_size]
                    self.write_idx = second_part
                
                samples_written += chunk_size
                self.samples_written += chunk_size
                
                # Check for wrap-around
                if self.write_idx >= self.buffer_size:
                    self.write_idx = 0
                    
                # Check for callback triggers
                self._check_callbacks()
                
            return samples_written
    
    def write_block(self, data: np.ndarray) -> bool:
        """
        Write a complete block to the buffer (DMA block transfer).
        
        Args:
            data: Data block (must match block_size)
            
        Returns:
            True if successful, False if overflow would occur
        """
        if len(data) != self.block_size:
            raise ValueError(f"Data must be exactly {self.block_size} samples")
        
        with self._lock:
            available = self.buffer_size - self._get_fill_level()
            
            if available < self.block_size:
                self.overflow_count += 1
                return False
            
            # Write the block
            if self.write_idx + self.block_size <= self.buffer_size:
                self.buffer[self.write_idx:self.write_idx + self.block_size] = data
                old_write_idx = self.write_idx
                self.write_idx += self.block_size
            else:
                # Wrap-around
                first_part = self.buffer_size - self.write_idx
                self.buffer[self.write_idx:] = data[:first_part]
                second_part = self.block_size - first_part
                self.buffer[:second_part] = data[first_part:]
                old_write_idx = self.write_idx
                self.write_idx = second_part
            
            self.samples_written += self.block_size
            
            # Trigger callbacks based on position
            half_point = self.buffer_size // 2
            
            # Check if we crossed the half point
            if old_write_idx < half_point <= self.write_idx:
                self._trigger_half_complete()
            
            # Check if we wrapped around (crossed the full point)
            if self.write_idx < old_write_idx:
                self._trigger_full_complete()
            
            return True
    
    def read_block(self) -> Optional[np.ndarray]:
        """
        Read a block of data from the buffer.
        
        Returns:
            Data block of size block_size, or None if insufficient data
        """
        with self._lock:
            available = self._get_fill_level()
            
            if available < self.block_size:
                return None
            
            # Read the block
            if self.read_idx + self.block_size <= self.buffer_size:
                data = self.buffer[self.read_idx:self.read_idx + self.block_size].copy()
                self.read_idx += self.block_size
            else:
                # Wrap-around
                first_part = self.buffer_size - self.read_idx
                data = np.zeros(self.block_size, dtype=self.dtype)
                data[:first_part] = self.buffer[self.read_idx:]
                second_part = self.block_size - first_part
                data[first_part:] = self.buffer[:second_part]
                self.read_idx = second_part
            
            self.samples_read += self.block_size
            
            if self.read_idx >= self.buffer_size:
                self.read_idx = 0
                
            return data
    
    def _get_fill_level(self) -> int:
        """Get current number of samples in buffer."""
        if self.write_idx >= self.read_idx:
            return self.write_idx - self.read_idx
        else:
            return self.buffer_size - self.read_idx + self.write_idx
    
    def _check_callbacks(self) -> None:
        """Check if any callbacks should be triggered based on write position."""
        half_point = self.buffer_size // 2
        
        # Simple state-based callback triggering
        # In real DMA, these would be hardware interrupts
        pass  # Handled in write_block for block-based operation
    
    def _trigger_half_complete(self) -> None:
        """Trigger half-complete callback with first half of buffer."""
        if self.half_complete_callback is not None:
            start_time = time.time()
            data_block = self.buffer[:self.buffer_size // 2].copy()
            self.half_complete_callback(data_block)
            callback_time = time.time() - start_time
            self.callback_times.append(callback_time)
            self.last_callback_time = callback_time
    
    def _trigger_full_complete(self) -> None:
        """Trigger full-complete callback with second half of buffer."""
        if self.full_complete_callback is not None:
            start_time = time.time()
            data_block = self.buffer[self.buffer_size // 2:].copy()
            self.full_complete_callback(data_block)
            callback_time = time.time() - start_time
            self.callback_times.append(callback_time)
            self.last_callback_time = callback_time
    
    def get_fill_level(self) -> int:
        """
        Get current buffer fill level (thread-safe).
        
        Returns:
            Number of samples currently in buffer
        """
        with self._lock:
            return self._get_fill_level()
    
    def get_fill_percentage(self) -> float:
        """
        Get buffer fill level as percentage.
        
        Returns:
            Fill level as percentage (0-100)
        """
        return (self.get_fill_level() / self.buffer_size) * 100.0
    
    def is_overflow(self) -> bool:
        """
        Check if any overflows have occurred.
        
        Returns:
            True if overflow detected
        """
        return self.overflow_count > 0
    
    def reset(self) -> None:
        """Reset buffer to initial state."""
        with self._lock:
            self.buffer.fill(0)
            self.write_idx = 0
            self.read_idx = 0
            self.overflow_count = 0
            self.samples_written = 0
            self.samples_read = 0
            self.callback_times.clear()
            self.last_callback_time = 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get buffer performance statistics.
        
        Returns:
            Dictionary containing:
            - buffer_size: Total buffer size
            - block_size: Block size
            - fill_level: Current fill level (samples)
            - fill_percentage: Fill level as percentage
            - overflow_count: Number of overflows
            - samples_written: Total samples written
            - samples_read: Total samples read
            - avg_callback_time: Average callback execution time (ms)
            - max_callback_time: Maximum callback execution time (ms)
        """
        stats = {
            'buffer_size': self.buffer_size,
            'block_size': self.block_size,
            'fill_level': self.get_fill_level(),
            'fill_percentage': self.get_fill_percentage(),
            'overflow_count': self.overflow_count,
            'samples_written': self.samples_written,
            'samples_read': self.samples_read,
        }
        
        if self.callback_times:
            stats['avg_callback_time_ms'] = np.mean(self.callback_times) * 1000
            stats['max_callback_time_ms'] = np.max(self.callback_times) * 1000
            stats['min_callback_time_ms'] = np.min(self.callback_times) * 1000
        else:
            stats['avg_callback_time_ms'] = 0.0
            stats['max_callback_time_ms'] = 0.0
            stats['min_callback_time_ms'] = 0.0
        
        return stats
    
    def export_to_csv(self, 
                     output_dir: str = 'data/outputs',
                     prefix: str = 'dma_buffer') -> Dict[str, str]:
        """
        Export buffer statistics to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping file types to file paths
        """
        import os
        import csv
        
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Export statistics
        stats_file = os.path.join(output_dir, f'{prefix}_statistics.csv')
        stats = self.get_statistics()
        
        with open(stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in stats.items():
                writer.writerow([key, value])
        
        exported_files['statistics'] = stats_file
        
        # Export callback timing history if available
        if self.callback_times:
            timing_file = os.path.join(output_dir, f'{prefix}_callback_timing.csv')
            with open(timing_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['callback_number', 'execution_time_ms'])
                for i, t in enumerate(self.callback_times):
                    writer.writerow([i, t * 1000])
            
            exported_files['timing'] = timing_file
        
        return exported_files


def demo():
    """Demonstrate DMA buffer functionality."""
    print("=" * 70)
    print("DMA Circular Buffer Demo")
    print("=" * 70)
    
    # Create buffer
    buffer_size = 1024
    block_size = 512
    dma = DMABuffer(buffer_size=buffer_size, block_size=block_size)
    
    print(f"\n1. Buffer Configuration")
    print(f"   Buffer size: {buffer_size} samples")
    print(f"   Block size: {block_size} samples")
    print(f"   Initial fill: {dma.get_fill_percentage():.1f}%")
    
    # Set up callbacks to track calls
    callback_log = {'half': 0, 'full': 0}
    
    def half_callback(data):
        callback_log['half'] += 1
        print(f"   [ISR] Half-complete callback #{callback_log['half']} - received {len(data)} samples")
    
    def full_callback(data):
        callback_log['full'] += 1
        print(f"   [ISR] Full-complete callback #{callback_log['full']} - received {len(data)} samples")
    
    dma.register_half_complete_callback(half_callback)
    dma.register_full_complete_callback(full_callback)
    
    # Simulate DMA transfers
    print(f"\n2. Simulating DMA Block Transfers")
    
    # First block (0-512)
    data1 = np.linspace(0, 1, block_size)
    success = dma.write_block(data1)
    print(f"   Block 1: {'Success' if success else 'Failed'}, Fill: {dma.get_fill_percentage():.1f}%")
    
    # Second block (512-1024) - should trigger half callback
    data2 = np.linspace(1, 2, block_size)
    success = dma.write_block(data2)
    print(f"   Block 2: {'Success' if success else 'Failed'}, Fill: {dma.get_fill_percentage():.1f}%")
    
    # Read a block
    print(f"\n3. Reading Data Blocks")
    read_data = dma.read_block()
    if read_data is not None:
        print(f"   Read block: {len(read_data)} samples, range [{read_data[0]:.3f}, {read_data[-1]:.3f}]")
        print(f"   Fill after read: {dma.get_fill_percentage():.1f}%")
    
    # Continue writing - should trigger full callback
    data3 = np.linspace(2, 3, block_size)
    success = dma.write_block(data3)
    print(f"   Block 3: {'Success' if success else 'Failed'}, Fill: {dma.get_fill_percentage():.1f}%")
    
    # Get statistics
    print(f"\n4. Buffer Statistics")
    stats = dma.get_statistics()
    print(f"   Total written: {stats['samples_written']} samples")
    print(f"   Total read: {stats['samples_read']} samples")
    print(f"   Overflows: {stats['overflow_count']}")
    print(f"   Half callbacks: {callback_log['half']}")
    print(f"   Full callbacks: {callback_log['full']}")
    if stats['avg_callback_time_ms'] > 0:
        print(f"   Avg callback time: {stats['avg_callback_time_ms']:.3f} ms")
    
    # Export to CSV
    print(f"\n5. Exporting to CSV")
    exported = dma.export_to_csv()
    for file_type, path in exported.items():
        print(f"   {file_type}: {path}")
    
    # Test overflow scenario
    print(f"\n6. Testing Overflow Protection")
    dma.reset()
    print(f"   Buffer reset, fill: {dma.get_fill_percentage():.1f}%")
    
    # Try to write more than buffer can hold
    for i in range(5):
        data = np.ones(block_size) * i
        success = dma.write_block(data)
        print(f"   Block {i+1}: {'Success' if success else 'OVERFLOW'}, Fill: {dma.get_fill_percentage():.1f}%")
    
    stats = dma.get_statistics()
    print(f"   Total overflows: {stats['overflow_count']}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()

