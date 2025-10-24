#!/usr/bin/env python3
"""
Image Generation Script for IEEE Report
Generates all required figures for the LDPC decoder implementation report
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
import time
import logging
from typing import Dict, List, Tuple
import os

# Import all modules
from ldpc_decoder import LDPCCode, BasicMinSumDecoder, create_test_ldpc_code, simulate_awgn_channel
from neural_minsum_decoder import NeuralMinSumDecoder, NeuralOffsetMinSumDecoder, analyze_weight_patterns
from neural_2d_decoder import Neural2DMinSumDecoder, Neural2DOffsetMinSumDecoder
from rcq_decoder import RCQMinSumDecoder, WeightedRCQDecoder, NonUniformQuantizer
from training_framework import TrainingConfig, PosteriorJointTrainer, GradientExplosionAnalyzer, create_dvbs2_code
from simulation_framework import SimulationConfig, LDPSimulator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class ImageGenerator:
    """Generate publication-quality images for IEEE report"""
    
    def __init__(self, output_dir: str = "images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette for different decoders
        self.colors = {
            'Basic MinSum': '#1f77b4',
            'N-NMS': '#ff7f0e', 
            'N-OMS': '#2ca02c',
            'N-2D-NMS Type 1': '#d62728',
            'N-2D-NMS Type 2': '#9467bd',
            'N-2D-NMS Type 3': '#8c564b',
            'N-2D-NMS Type 4': '#e377c2',
            'RCQ MinSum': '#7f7f7f',
            'W-RCQ Type 2': '#bcbd22'
        }
        
        self.markers = {
            'Basic MinSum': 'o',
            'N-NMS': 's',
            'N-OMS': '^',
            'N-2D-NMS Type 1': 'v',
            'N-2D-NMS Type 2': '<',
            'N-2D-NMS Type 3': '>',
            'N-2D-NMS Type 4': 'p',
            'RCQ MinSum': '*',
            'W-RCQ Type 2': 'h'
        }
    
    def generate_performance_data(self, code: LDPCCode, snr_range: Tuple[float, float] = (0.0, 6.0), 
                                snr_step: float = 0.5, max_frames: int = 1000) -> Dict:
        """Generate performance data for all decoders"""
        logger.info("Generating performance data for all decoders...")
        
        # Create decoders
        decoders = {
            'Basic MinSum': BasicMinSumDecoder(code, factor=0.7),
            'N-NMS': NeuralMinSumDecoder(code, max_iterations=10),
            'N-OMS': NeuralOffsetMinSumDecoder(code, max_iterations=10),
            'N-2D-NMS Type 1': Neural2DMinSumDecoder(code, weight_sharing_type=1, max_iterations=10),
            'N-2D-NMS Type 2': Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=10),
            'N-2D-NMS Type 3': Neural2DMinSumDecoder(code, weight_sharing_type=3, max_iterations=10),
            'N-2D-NMS Type 4': Neural2DMinSumDecoder(code, weight_sharing_type=4, max_iterations=10),
            'RCQ MinSum': RCQMinSumDecoder(code, bc=3, bv=8, quantizer_params=[(3.0, 1.3), (5.0, 1.3), (7.0, 1.3)], max_iterations=10),
            'W-RCQ Type 2': WeightedRCQDecoder(code, bc=3, bv=8, quantizer_params=[(3.0, 1.3), (5.0, 1.3), (7.0, 1.3)], weight_sharing_type=2, max_iterations=10)
        }
        
        # Generate SNR values
        snr_values = np.arange(snr_range[0], snr_range[1] + snr_step, snr_step)
        
        # Test codeword
        test_codeword = np.zeros(code.n, dtype=int)
        
        results = {}
        
        for decoder_name, decoder in decoders.items():
            logger.info(f"Testing {decoder_name}...")
            
            fers = []
            bers = []
            avg_iters = []
            
            for snr_db in snr_values:
                frame_errors = 0
                bit_errors = 0
                total_iterations = 0
                total_frames = 0
                
                # Simulate multiple frames
                for _ in range(max_frames):
                    # Generate test data
                    llr = simulate_awgn_channel(test_codeword, snr_db)
                    
                    # Decode
                    if isinstance(decoder, torch.nn.Module):
                        llr_tensor = torch.tensor(llr, dtype=torch.float32)
                        decoded, posterior, iterations = decoder(llr_tensor)
                        decoded = decoded.numpy()
                    else:
                        decoded, success, iterations = decoder.decode(llr)
                    
                    # Count errors
                    frame_error = not np.array_equal(decoded, test_codeword)
                    if frame_error:
                        frame_errors += 1
                        bit_errors += np.sum(decoded != test_codeword)
                    
                    total_iterations += iterations
                    total_frames += 1
                
                # Calculate metrics
                fer = frame_errors / total_frames if total_frames > 0 else 0.0
                ber = bit_errors / (total_frames * code.n) if total_frames > 0 else 0.0
                avg_iter = total_iterations / total_frames if total_frames > 0 else 0.0
                
                fers.append(fer)
                bers.append(ber)
                avg_iters.append(avg_iter)
            
            results[decoder_name] = {
                'snr': snr_values,
                'fer': np.array(fers),
                'ber': np.array(bers),
                'avg_iterations': np.array(avg_iters)
            }
        
        return results
    
    def generate_fer_comparison(self, results: Dict, save_path: str = None):
        """Generate FER comparison plot"""
        logger.info("Generating FER comparison plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for decoder_name, data in results.items():
            # Avoid plotting zero FER values
            fer_data = data['fer']
            snr_data = data['snr']
            
            # Replace zeros with small values for log scale
            fer_data = np.where(fer_data == 0, 1e-6, fer_data)
            
            ax.semilogy(snr_data, fer_data, 
                       marker=self.markers[decoder_name],
                       color=self.colors[decoder_name],
                       label=decoder_name,
                       linewidth=2,
                       markersize=8,
                       markevery=2)
        
        ax.set_xlabel('SNR (dB)', fontsize=14)
        ax.set_ylabel('Frame Error Rate (FER)', fontsize=14)
        ax.set_title('Frame Error Rate vs SNR', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set axis limits
        ax.set_xlim(0, 6)
        ax.set_ylim(1e-6, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"FER plot saved to {save_path}")
        
        plt.show()
    
    def generate_ber_comparison(self, results: Dict, save_path: str = None):
        """Generate BER comparison plot"""
        logger.info("Generating BER comparison plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for decoder_name, data in results.items():
            # Avoid plotting zero BER values
            ber_data = data['ber']
            snr_data = data['snr']
            
            # Replace zeros with small values for log scale
            ber_data = np.where(ber_data == 0, 1e-8, ber_data)
            
            ax.semilogy(snr_data, ber_data, 
                       marker=self.markers[decoder_name],
                       color=self.colors[decoder_name],
                       label=decoder_name,
                       linewidth=2,
                       markersize=8,
                       markevery=2)
        
        ax.set_xlabel('SNR (dB)', fontsize=14)
        ax.set_ylabel('Bit Error Rate (BER)', fontsize=14)
        ax.set_title('Bit Error Rate vs SNR', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set axis limits
        ax.set_xlim(0, 6)
        ax.set_ylim(1e-8, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"BER plot saved to {save_path}")
        
        plt.show()
    
    def generate_gradient_analysis(self, code: LDPCCode, save_path: str = None):
        """Generate gradient explosion analysis plot"""
        logger.info("Generating gradient analysis plot...")
        
        # Create models for analysis
        model_standard = NeuralMinSumDecoder(code, max_iterations=20)
        model_posterior = NeuralMinSumDecoder(code, max_iterations=20)
        
        # Analyze gradient explosion
        analyzer_standard = GradientExplosionAnalyzer(model_standard, code)
        analyzer_posterior = GradientExplosionAnalyzer(model_posterior, code)
        
        # Generate gradient data
        grad_results_standard = analyzer_standard.analyze_gradient_explosion(num_samples=50)
        grad_results_posterior = analyzer_posterior.analyze_gradient_explosion(num_samples=50)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Gradient magnitude distribution
        ax1.hist(grad_results_standard['gradient_magnitudes'], bins=20, alpha=0.7, 
                color='red', label='Standard Training', density=True)
        ax1.hist(grad_results_posterior['gradient_magnitudes'], bins=20, alpha=0.7, 
                color='blue', label='Posterior Joint Training', density=True)
        ax1.set_xlabel('Gradient Magnitude', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Gradient Magnitude Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gradient vs iterations
        ax2.scatter(grad_results_standard['iteration_counts'], grad_results_standard['gradient_magnitudes'], 
                   alpha=0.6, color='red', label='Standard Training', s=50)
        ax2.scatter(grad_results_posterior['iteration_counts'], grad_results_posterior['gradient_magnitudes'], 
                   alpha=0.6, color='blue', label='Posterior Joint Training', s=50)
        ax2.set_xlabel('Iterations', fontsize=12)
        ax2.set_ylabel('Gradient Magnitude', fontsize=12)
        ax2.set_title('Gradient Magnitude vs Iterations', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Standard Training:
Mean: {grad_results_standard['mean_gradient']:.4f}
Max: {grad_results_standard['max_gradient']:.4f}

Posterior Training:
Mean: {grad_results_posterior['mean_gradient']:.4f}
Max: {grad_results_posterior['max_gradient']:.4f}"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gradient analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_parameter_comparison(self, code: LDPCCode, save_path: str = None):
        """Generate parameter comparison plot"""
        logger.info("Generating parameter comparison plot...")
        
        # Calculate parameters for different decoders
        decoders_info = {
            'N-NMS': NeuralMinSumDecoder(code, max_iterations=10),
            'N-2D-NMS Type 1': Neural2DMinSumDecoder(code, weight_sharing_type=1, max_iterations=10),
            'N-2D-NMS Type 2': Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=10),
            'N-2D-NMS Type 3': Neural2DMinSumDecoder(code, weight_sharing_type=3, max_iterations=10),
            'N-2D-NMS Type 4': Neural2DMinSumDecoder(code, weight_sharing_type=4, max_iterations=10),
        }
        
        decoder_names = []
        parameters = []
        reduction_factors = []
        
        baseline_params = len(decoders_info['N-NMS'].beta_weights)
        
        for name, decoder in decoders_info.items():
            if hasattr(decoder, 'beta_weights') and hasattr(decoder, 'alpha_weights'):
                num_params = len(decoder.beta_weights) + len(decoder.alpha_weights)
            else:
                num_params = len(decoder.beta_weights)
            
            decoder_names.append(name)
            parameters.append(num_params)
            reduction_factors.append(baseline_params / num_params)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Parameter count
        bars1 = ax1.bar(range(len(decoder_names)), parameters, 
                       color=[self.colors.get(name, '#666666') for name in decoder_names])
        ax1.set_xlabel('Decoder Type', fontsize=12)
        ax1.set_ylabel('Number of Parameters', fontsize=12)
        ax1.set_title('Parameter Count Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(decoder_names)))
        ax1.set_xticklabels(decoder_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, param) in enumerate(zip(bars1, parameters)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{param}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Reduction factor
        bars2 = ax2.bar(range(len(decoder_names)), reduction_factors, 
                       color=[self.colors.get(name, '#666666') for name in decoder_names])
        ax2.set_xlabel('Decoder Type', fontsize=12)
        ax2.set_ylabel('Parameter Reduction Factor', fontsize=12)
        ax2.set_title('Parameter Reduction Factor', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(decoder_names)))
        ax2.set_xticklabels(decoder_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, reduction) in enumerate(zip(bars2, reduction_factors)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{reduction:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_weight_pattern_analysis(self, code: LDPCCode, save_path: str = None):
        """Generate weight pattern analysis plot"""
        logger.info("Generating weight pattern analysis plot...")
        
        decoder = NeuralMinSumDecoder(code, max_iterations=10)
        analysis = analyze_weight_patterns(decoder, code)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Weight evolution over iterations
        iterations = list(analysis['iteration_patterns'].keys())
        means = [analysis['iteration_patterns'][t]['mean'] for t in iterations]
        stds = [analysis['iteration_patterns'][t]['std'] for t in iterations]
        
        ax1.errorbar(iterations, means, yerr=stds, marker='o', linewidth=2, markersize=8,
                    color='blue', capsize=5, capthick=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Mean Weight Value', fontsize=12)
        ax1.set_title('Weight Evolution Over Iterations', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Weight distribution by check node degree
        degrees = list(analysis['node_degree_correlations'].keys())
        degree_means = [analysis['node_degree_correlations'][d]['mean'] for d in degrees]
        degree_stds = [analysis['node_degree_correlations'][d]['std'] for d in degrees]
        degree_counts = [analysis['node_degree_correlations'][d]['count'] for d in degrees]
        
        bars = ax2.bar(range(len(degrees)), degree_means, 
                      yerr=degree_stds, capsize=5, capthick=2,
                      color='green', alpha=0.7)
        ax2.set_xlabel('Check Node Degree', fontsize=12)
        ax2.set_ylabel('Mean Weight Value', fontsize=12)
        ax2.set_title('Weight Distribution by Check Node Degree', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(degrees)))
        ax2.set_xticklabels([d.replace('check_degree_', '') for d in degrees])
        ax2.grid(True, alpha=0.3)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, degree_counts)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weight pattern analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_all_images(self, code: LDPCCode = None):
        """Generate all required images for the IEEE report"""
        if code is None:
            code = create_test_ldpc_code()
        
        logger.info("Starting image generation for IEEE report...")
        
        # Generate performance data
        logger.info("Generating performance data...")
        results = self.generate_performance_data(code, snr_range=(0.0, 6.0), snr_step=0.5, max_frames=500)
        
        # Generate all plots
        logger.info("Generating FER comparison plot...")
        self.generate_fer_comparison(results, os.path.join(self.output_dir, 'fer_comparison.png'))
        
        logger.info("Generating BER comparison plot...")
        self.generate_ber_comparison(results, os.path.join(self.output_dir, 'ber_comparison.png'))
        
        logger.info("Generating gradient analysis plot...")
        self.generate_gradient_analysis(code, os.path.join(self.output_dir, 'gradient_analysis.png'))
        
        logger.info("Generating parameter comparison plot...")
        self.generate_parameter_comparison(code, os.path.join(self.output_dir, 'parameter_comparison.png'))
        
        logger.info("Generating weight pattern analysis plot...")
        self.generate_weight_pattern_analysis(code, os.path.join(self.output_dir, 'weight_pattern_analysis.png'))
        
        logger.info("All images generated successfully!")
        logger.info(f"Images saved in directory: {self.output_dir}")

def main():
    """Main function to generate all images"""
    print("=" * 80)
    print("IEEE REPORT IMAGE GENERATION")
    print("=" * 80)
    
    # Create image generator
    generator = ImageGenerator(output_dir="images")
    
    # Create test code
    code = create_test_ldpc_code()
    print(f"Using LDPC code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    
    # Generate all images
    generator.generate_all_images(code)
    
    print("\n" + "=" * 80)
    print("IMAGE GENERATION COMPLETED!")
    print("=" * 80)
    print("Generated images:")
    print("1. fer_comparison.png - Frame Error Rate comparison")
    print("2. ber_comparison.png - Bit Error Rate comparison") 
    print("3. gradient_analysis.png - Gradient explosion analysis")
    print("4. parameter_comparison.png - Parameter reduction comparison")
    print("5. weight_pattern_analysis.png - Weight pattern analysis")
    print("\nThese images can now be used in the IEEE LaTeX report!")

if __name__ == "__main__":
    main()
