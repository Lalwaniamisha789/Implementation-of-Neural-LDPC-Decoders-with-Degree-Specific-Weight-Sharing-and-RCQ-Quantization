"""
Simulation Framework for LDPC Decoder Performance Evaluation
Based on the paper: arXiv:2310.15483v2

This module provides comprehensive simulation capabilities for evaluating
different LDPC decoders including FER curves, BER curves, and performance comparisons.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from ldpc_decoder import LDPCCode, BasicMinSumDecoder, simulate_awgn_channel
from neural_minsum_decoder import NeuralMinSumDecoder, NeuralOffsetMinSumDecoder
from neural_2d_decoder import Neural2DMinSumDecoder, Neural2DOffsetMinSumDecoder
from rcq_decoder import RCQMinSumDecoder, WeightedRCQDecoder
from training_framework import TrainingConfig, PosteriorJointTrainer

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Simulation configuration"""
    snr_range: Tuple[float, float] = (0.0, 6.0)
    snr_step: float = 0.5
    max_frames: int = 10000
    max_errors: int = 100
    min_frames: int = 1000
    parallel_workers: int = 4
    device: str = 'cpu'
    save_results: bool = True
    results_dir: str = 'simulation_results'

class SimulationResult:
    """Container for simulation results"""
    
    def __init__(self, decoder_name: str, snr_values: List[float]):
        self.decoder_name = decoder_name
        self.snr_values = snr_values
        self.frame_error_rates = []
        self.bit_error_rates = []
        self.average_iterations = []
        self.simulation_times = []
        self.total_frames = []
        self.total_errors = []
    
    def add_result(self, snr_idx: int, fer: float, ber: float, 
                   avg_iter: float, sim_time: float, total_frames: int, total_errors: int):
        """Add result for a specific SNR"""
        while len(self.frame_error_rates) <= snr_idx:
            self.frame_error_rates.append(0.0)
            self.bit_error_rates.append(0.0)
            self.average_iterations.append(0.0)
            self.simulation_times.append(0.0)
            self.total_frames.append(0)
            self.total_errors.append(0)
        
        self.frame_error_rates[snr_idx] = fer
        self.bit_error_rates[snr_idx] = ber
        self.average_iterations[snr_idx] = avg_iter
        self.simulation_times[snr_idx] = sim_time
        self.total_frames[snr_idx] = total_frames
        self.total_errors[snr_idx] = total_errors

class LDPSimulator:
    """
    Comprehensive LDPC decoder simulator
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results: Dict[str, SimulationResult] = {}
        
        # Create results directory if needed
        if config.save_results:
            import os
            os.makedirs(config.results_dir, exist_ok=True)
    
    def simulate_single_snr(self, decoder: Callable, code: LDPCCode, 
                           snr_db: float, max_frames: int, max_errors: int) -> Tuple[float, float, float, float, int, int]:
        """
        Simulate decoder performance at a single SNR
        
        Args:
            decoder: Decoder function
            code: LDPC code
            snr_db: SNR in dB
            max_frames: Maximum number of frames to simulate
            max_errors: Maximum number of errors to collect
            
        Returns:
            fer, ber, avg_iterations, simulation_time, total_frames, total_errors
        """
        start_time = time.time()
        
        frame_errors = 0
        bit_errors = 0
        total_iterations = 0
        total_frames = 0
        
        # Generate all-zero codeword
        codeword = np.zeros(code.n, dtype=int)
        
        while total_frames < max_frames and frame_errors < max_errors:
            # Simulate AWGN channel
            llr = simulate_awgn_channel(codeword, snr_db)
            
            # Decode
            if isinstance(decoder, torch.nn.Module):
                # Neural decoder
                llr_tensor = torch.tensor(llr, dtype=torch.float32)
                decoded, posterior, iterations = decoder(llr_tensor)
                decoded = decoded.numpy()
            else:
                # Basic decoder
                decoded, success, iterations = decoder.decode(llr)
            
            # Count errors
            frame_error = not np.array_equal(decoded, codeword)
            if frame_error:
                frame_errors += 1
                bit_errors += np.sum(decoded != codeword)
            
            total_iterations += iterations
            total_frames += 1
        
        # Calculate metrics
        fer = frame_errors / total_frames if total_frames > 0 else 0.0
        ber = bit_errors / (total_frames * code.n) if total_frames > 0 else 0.0
        avg_iterations = total_iterations / total_frames if total_frames > 0 else 0.0
        simulation_time = time.time() - start_time
        
        return fer, ber, avg_iterations, simulation_time, total_frames, frame_errors
    
    def simulate_decoder(self, decoder: Union[Callable, torch.nn.Module], 
                        code: LDPCCode, decoder_name: str) -> SimulationResult:
        """
        Simulate decoder performance across SNR range
        
        Args:
            decoder: Decoder to simulate
            code: LDPC code
            decoder_name: Name for the decoder
            
        Returns:
            Simulation results
        """
        logger.info(f"Starting simulation for {decoder_name}")
        
        # Generate SNR values
        snr_values = np.arange(self.config.snr_range[0], 
                              self.config.snr_range[1] + self.config.snr_step, 
                              self.config.snr_step)
        
        result = SimulationResult(decoder_name, snr_values.tolist())
        
        for snr_idx, snr_db in enumerate(snr_values):
            logger.info(f"Simulating {decoder_name} at SNR = {snr_db:.1f} dB")
            
            fer, ber, avg_iter, sim_time, total_frames, total_errors = self.simulate_single_snr(
                decoder, code, snr_db, self.config.max_frames, self.config.max_errors
            )
            
            result.add_result(snr_idx, fer, ber, avg_iter, sim_time, total_frames, total_errors)
            
            logger.info(f"SNR {snr_db:.1f}dB: FER={fer:.2e}, BER={ber:.2e}, "
                       f"Avg Iter={avg_iter:.1f}, Time={sim_time:.1f}s")
        
        self.results[decoder_name] = result
        return result
    
    def simulate_multiple_decoders(self, decoders: Dict[str, Union[Callable, torch.nn.Module]], 
                                 code: LDPCCode) -> Dict[str, SimulationResult]:
        """
        Simulate multiple decoders in parallel
        
        Args:
            decoders: Dictionary of decoder names and decoder objects
            code: LDPC code
            
        Returns:
            Dictionary of simulation results
        """
        logger.info(f"Starting simulation for {len(decoders)} decoders")
        
        if self.config.parallel_workers > 1:
            # Parallel simulation
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = {}
                for name, decoder in decoders.items():
                    future = executor.submit(self.simulate_decoder, decoder, code, name)
                    futures[future] = name
                
                results = {}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        results[name] = result
                        logger.info(f"Completed simulation for {name}")
                    except Exception as e:
                        logger.error(f"Error simulating {name}: {e}")
        else:
            # Sequential simulation
            results = {}
            for name, decoder in decoders.items():
                result = self.simulate_decoder(decoder, code, name)
                results[name] = result
        
        return results
    
    def plot_fer_curves(self, results: Dict[str, SimulationResult], 
                       save_path: Optional[str] = None, log_scale: bool = True):
        """Plot Frame Error Rate curves"""
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            plt.semilogy(result.snr_values, result.frame_error_rates, 
                        marker='o', label=name, linewidth=2, markersize=6)
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('Frame Error Rate (FER)')
        plt.title('Frame Error Rate vs SNR')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ber_curves(self, results: Dict[str, SimulationResult], 
                       save_path: Optional[str] = None, log_scale: bool = True):
        """Plot Bit Error Rate curves"""
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            plt.semilogy(result.snr_values, result.bit_error_rates, 
                        marker='s', label=name, linewidth=2, markersize=6)
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('Bit Error Rate vs SNR')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_iteration_curves(self, results: Dict[str, SimulationResult], 
                             save_path: Optional[str] = None):
        """Plot average iteration curves"""
        plt.figure(figsize=(10, 8))
        
        for name, result in results.items():
            plt.plot(result.snr_values, result.average_iterations, 
                    marker='^', label=name, linewidth=2, markersize=6)
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('Average Iterations')
        plt.title('Average Iterations vs SNR')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comprehensive_comparison(self, results: Dict[str, SimulationResult], 
                                    save_path: Optional[str] = None):
        """Plot comprehensive comparison of all metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # FER curves
        for name, result in results.items():
            axes[0, 0].semilogy(result.snr_values, result.frame_error_rates, 
                              marker='o', label=name, linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('SNR (dB)')
        axes[0, 0].set_ylabel('Frame Error Rate')
        axes[0, 0].set_title('Frame Error Rate vs SNR')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # BER curves
        for name, result in results.items():
            axes[0, 1].semilogy(result.snr_values, result.bit_error_rates, 
                               marker='s', label=name, linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('SNR (dB)')
        axes[0, 1].set_ylabel('Bit Error Rate')
        axes[0, 1].set_title('Bit Error Rate vs SNR')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Average iterations
        for name, result in results.items():
            axes[1, 0].plot(result.snr_values, result.average_iterations, 
                            marker='^', label=name, linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('SNR (dB)')
        axes[1, 0].set_ylabel('Average Iterations')
        axes[1, 0].set_title('Average Iterations vs SNR')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Simulation time
        for name, result in results.items():
            axes[1, 1].plot(result.snr_values, result.simulation_times, 
                           marker='d', label=name, linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('SNR (dB)')
        axes[1, 1].set_ylabel('Simulation Time (s)')
        axes[1, 1].set_title('Simulation Time vs SNR')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results: Dict[str, SimulationResult], filename: str):
        """Save simulation results to file"""
        import json
        
        # Convert results to serializable format
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'decoder_name': result.decoder_name,
                'snr_values': result.snr_values,
                'frame_error_rates': result.frame_error_rates,
                'bit_error_rates': result.bit_error_rates,
                'average_iterations': result.average_iterations,
                'simulation_times': result.simulation_times,
                'total_frames': result.total_frames,
                'total_errors': result.total_errors
            }
        
        filepath = f"{self.config.results_dir}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filename: str) -> Dict[str, SimulationResult]:
        """Load simulation results from file"""
        import json
        
        filepath = f"{self.config.results_dir}/{filename}"
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = {}
        for name, result_data in data.items():
            result = SimulationResult(result_data['decoder_name'], result_data['snr_values'])
            result.frame_error_rates = result_data['frame_error_rates']
            result.bit_error_rates = result_data['bit_error_rates']
            result.average_iterations = result_data['average_iterations']
            result.simulation_times = result_data['simulation_times']
            result.total_frames = result_data['total_frames']
            result.total_errors = result_data['total_errors']
            results[name] = result
        
        logger.info(f"Results loaded from {filepath}")
        return results

def create_test_decoders(code: LDPCCode) -> Dict[str, Union[Callable, torch.nn.Module]]:
    """Create a set of test decoders for comparison"""
    decoders = {}
    
    # Basic MinSum decoder
    decoders['Basic MinSum'] = BasicMinSumDecoder(code, factor=0.7)
    
    # Neural MinSum decoder with edge-specific weights
    decoders['N-NMS'] = NeuralMinSumDecoder(code, max_iterations=10)
    
    # Neural Offset MinSum decoder
    decoders['N-OMS'] = NeuralOffsetMinSumDecoder(code, max_iterations=10)
    
    # Neural 2D MinSum decoders with different weight sharing types
    for weight_type in [1, 2, 3, 4]:
        decoders[f'N-2D-NMS Type {weight_type}'] = Neural2DMinSumDecoder(
            code, weight_sharing_type=weight_type, max_iterations=10
        )
    
    # Neural 2D Offset MinSum decoder
    decoders['N-2D-OMS Type 2'] = Neural2DOffsetMinSumDecoder(
        code, weight_sharing_type=2, max_iterations=10
    )
    
    # RCQ decoders
    quantizer_params = [(3.0, 1.3), (5.0, 1.3), (7.0, 1.3)]
    decoders['RCQ MinSum'] = RCQMinSumDecoder(
        code, bc=3, bv=8, quantizer_params=quantizer_params, max_iterations=10
    )
    
    # Weighted RCQ decoder
    decoders['W-RCQ Type 2'] = WeightedRCQDecoder(
        code, bc=3, bv=8, quantizer_params=quantizer_params, 
        weight_sharing_type=2, max_iterations=10
    )
    
    return decoders

if __name__ == "__main__":
    # Test simulation framework
    print("Testing LDPC Decoder Simulation Framework")
    
    # Create test code
    from training_framework import create_dvbs2_code
    code = create_dvbs2_code()
    print(f"Created LDPC code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    
    # Simulation configuration
    config = SimulationConfig(
        snr_range=(0.0, 4.0),
        snr_step=1.0,
        max_frames=1000,
        max_errors=50,
        parallel_workers=2,
        device='cpu'
    )
    
    # Create simulator
    simulator = LDPSimulator(config)
    
    # Create test decoders
    decoders = create_test_decoders(code)
    print(f"Created {len(decoders)} decoders for testing")
    
    # Run simulation
    print("Starting simulation...")
    results = simulator.simulate_multiple_decoders(decoders, code)
    
    # Plot results
    print("Plotting results...")
    simulator.plot_comprehensive_comparison(results)
    
    # Save results
    if config.save_results:
        simulator.save_results(results, 'test_simulation_results.json')
    
    print("Simulation completed!")
