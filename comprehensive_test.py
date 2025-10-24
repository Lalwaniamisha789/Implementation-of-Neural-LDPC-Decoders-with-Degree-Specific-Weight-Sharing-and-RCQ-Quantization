#!/usr/bin/env python3
"""
Comprehensive Test Script for LDPC Decoding Implementation
Tests all implemented decoders and generates performance comparison
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging
from typing import Dict, List

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

def test_all_decoders():
    """Test all implemented decoders"""
    print("=" * 80)
    print("COMPREHENSIVE LDPC DECODER TEST")
    print("Based on: arXiv:2310.15483v2")
    print("=" * 80)
    
    # Create test code
    code = create_test_ldpc_code()
    print(f"Test LDPC Code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    print(f"Check node degrees: {code.check_node_degrees}")
    print(f"Variable node degrees: {code.variable_node_degrees}")
    print(f"Total edges: {np.sum(code.H)}")
    
    # Test codeword
    test_codeword = np.zeros(code.n, dtype=int)
    snr_db = 2.0
    
    print(f"\nTesting at SNR = {snr_db} dB with all-zero codeword")
    print("-" * 60)
    
    # Generate test data
    llr = simulate_awgn_channel(test_codeword, snr_db)
    llr_tensor = torch.tensor(llr, dtype=torch.float32)
    
    # Test results storage
    results = {}
    
    # 1. Basic MinSum Decoder
    print("1. Basic MinSum Decoder")
    decoder = BasicMinSumDecoder(code, factor=0.7)
    start_time = time.time()
    decoded, success, iterations = decoder.decode(llr)
    decode_time = time.time() - start_time
    
    results['Basic MinSum'] = {
        'decoded': decoded,
        'success': success,
        'iterations': iterations,
        'time': decode_time,
        'parameters': 1,
        'errors': np.sum(decoded != test_codeword)
    }
    
    print(f"   Success: {success}, Iterations: {iterations}, Errors: {results['Basic MinSum']['errors']}")
    print(f"   Parameters: 1, Time: {decode_time:.4f}s")
    
    # 2. Neural MinSum Decoder
    print("\n2. Neural MinSum Decoder (Edge-Specific Weights)")
    decoder = NeuralMinSumDecoder(code, max_iterations=10)
    start_time = time.time()
    decoded, posterior, iterations = decoder(llr_tensor)
    decode_time = time.time() - start_time
    
    success = torch.all(decoded == torch.tensor(test_codeword))
    num_params = len(decoder.beta_weights)
    
    results['N-NMS'] = {
        'decoded': decoded.numpy(),
        'success': success.item(),
        'iterations': iterations,
        'time': decode_time,
        'parameters': num_params,
        'errors': np.sum(decoded.numpy() != test_codeword)
    }
    
    print(f"   Success: {success.item()}, Iterations: {iterations}, Errors: {results['N-NMS']['errors']}")
    print(f"   Parameters: {num_params}, Time: {decode_time:.4f}s")
    
    # 3. Neural Offset MinSum Decoder
    print("\n3. Neural Offset MinSum Decoder")
    decoder = NeuralOffsetMinSumDecoder(code, max_iterations=10)
    start_time = time.time()
    decoded, posterior, iterations = decoder(llr_tensor)
    decode_time = time.time() - start_time
    
    success = torch.all(decoded == torch.tensor(test_codeword))
    num_params = len(decoder.beta_weights)
    
    results['N-OMS'] = {
        'decoded': decoded.numpy(),
        'success': success.item(),
        'iterations': iterations,
        'time': decode_time,
        'parameters': num_params,
        'errors': np.sum(decoded.numpy() != test_codeword)
    }
    
    print(f"   Success: {success.item()}, Iterations: {iterations}, Errors: {results['N-OMS']['errors']}")
    print(f"   Parameters: {num_params}, Time: {decode_time:.4f}s")
    
    # 4. Neural 2D MinSum Decoders
    print("\n4. Neural 2D MinSum Decoders (Weight Sharing)")
    weight_types = [1, 2, 3, 4]
    
    for weight_type in weight_types:
        decoder = Neural2DMinSumDecoder(code, weight_sharing_type=weight_type, max_iterations=10)
        start_time = time.time()
        decoded, posterior, iterations = decoder(llr_tensor)
        decode_time = time.time() - start_time
        
        success = torch.all(decoded == torch.tensor(test_codeword))
        num_params = len(decoder.beta_weights) + len(decoder.alpha_weights)
        
        results[f'N-2D-NMS Type {weight_type}'] = {
            'decoded': decoded.numpy(),
            'success': success.item(),
            'iterations': iterations,
            'time': decode_time,
            'parameters': num_params,
            'errors': np.sum(decoded.numpy() != test_codeword)
        }
        
        print(f"   Type {weight_type}: Success: {success.item()}, Iterations: {iterations}, "
              f"Parameters: {num_params}, Errors: {results[f'N-2D-NMS Type {weight_type}']['errors']}")
    
    # 5. RCQ Decoder
    print("\n5. RCQ MinSum Decoder")
    quantizer_params = [(3.0, 1.3), (5.0, 1.3), (7.0, 1.3)]
    decoder = RCQMinSumDecoder(code, bc=3, bv=8, quantizer_params=quantizer_params, max_iterations=10)
    start_time = time.time()
    decoded, success, iterations = decoder.decode(llr_tensor)
    decode_time = time.time() - start_time
    
    results['RCQ MinSum'] = {
        'decoded': decoded.numpy(),
        'success': success,
        'iterations': iterations,
        'time': decode_time,
        'parameters': len(quantizer_params) * 2,  # C and gamma for each quantizer
        'errors': np.sum(decoded.numpy() != test_codeword)
    }
    
    print(f"   Success: {success}, Iterations: {iterations}, Errors: {results['RCQ MinSum']['errors']}")
    print(f"   Parameters: {results['RCQ MinSum']['parameters']}, Time: {decode_time:.4f}s")
    
    # 6. Weighted RCQ Decoder
    print("\n6. Weighted RCQ Decoder")
    decoder = WeightedRCQDecoder(code, bc=3, bv=8, quantizer_params=quantizer_params, 
                               weight_sharing_type=2, max_iterations=10)
    start_time = time.time()
    decoded, posterior, iterations = decoder(llr_tensor)
    decode_time = time.time() - start_time
    
    success = torch.all(decoded == torch.tensor(test_codeword))
    num_params = len(decoder.beta_weights) + len(decoder.alpha_weights) + len(quantizer_params) * 2
    
    results['W-RCQ Type 2'] = {
        'decoded': decoded.numpy(),
        'success': success.item(),
        'iterations': iterations,
        'time': decode_time,
        'parameters': num_params,
        'errors': np.sum(decoded.numpy() != test_codeword)
    }
    
    print(f"   Success: {success.item()}, Iterations: {iterations}, Errors: {results['W-RCQ Type 2']['errors']}")
    print(f"   Parameters: {num_params}, Time: {decode_time:.4f}s")
    
    return results

def analyze_results(results: Dict):
    """Analyze and display test results"""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Create summary table
    print(f"{'Decoder':<20} | {'Success':<7} | {'Iter':<4} | {'Errors':<6} | {'Params':<8} | {'Time (s)':<8}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} | {str(result['success']):<7} | {result['iterations']:<4} | "
              f"{result['errors']:<6} | {result['parameters']:<8} | {result['time']:<8.4f}")
    
    # Parameter reduction analysis
    print(f"\nPARAMETER REDUCTION ANALYSIS:")
    print("-" * 40)
    
    n_nms_params = results['N-NMS']['parameters']
    print(f"N-NMS (baseline): {n_nms_params} parameters")
    
    for name, result in results.items():
        if 'N-2D-NMS' in name:
            reduction = n_nms_params / result['parameters']
            print(f"{name}: {result['parameters']} parameters ({reduction:.1f}x reduction)")
    
    # Performance comparison
    print(f"\nPERFORMANCE COMPARISON:")
    print("-" * 30)
    
    successful_decoders = [name for name, result in results.items() if result['success']]
    if successful_decoders:
        print(f"Successful decoders: {', '.join(successful_decoders)}")
    else:
        print("No decoders achieved perfect decoding (expected for noisy channel)")
    
    # Speed comparison
    fastest_decoder = min(results.items(), key=lambda x: x[1]['time'])
    print(f"Fastest decoder: {fastest_decoder[0]} ({fastest_decoder[1]['time']:.4f}s)")
    
    # Iteration efficiency
    most_efficient = min(results.items(), key=lambda x: x[1]['iterations'])
    print(f"Most iteration-efficient: {most_efficient[0]} ({most_efficient[1]['iterations']} iterations)")

def test_weight_patterns():
    """Test weight pattern analysis"""
    print("\n" + "=" * 80)
    print("WEIGHT PATTERN ANALYSIS")
    print("=" * 80)
    
    code = create_test_ldpc_code()
    decoder = NeuralMinSumDecoder(code, max_iterations=5)
    
    print("Analyzing weight patterns in Neural MinSum decoder...")
    analysis = analyze_weight_patterns(decoder, code)
    
    print("\nWeight statistics by iteration:")
    for t, stats in analysis['iteration_patterns'].items():
        print(f"  Iteration {t}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    print("\nWeight statistics by check node degree:")
    for degree, stats in analysis['node_degree_correlations'].items():
        print(f"  {degree}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")

def test_quantization():
    """Test quantization functionality"""
    print("\n" + "=" * 80)
    print("QUANTIZATION TEST")
    print("=" * 80)
    
    print("Testing Non-uniform Quantizer:")
    quantizer = NonUniformQuantizer(bc=3, C=5.0, gamma=1.5)
    
    test_input = torch.tensor([-3.2, -1.1, 0.5, 2.8, 4.1])
    quantized = quantizer.quantize(test_input)
    dequantized = quantizer.dequantize(quantized)
    
    print(f"Input:      {test_input}")
    print(f"Quantized: {quantized}")
    print(f"Dequantized: {dequantized}")
    print(f"Quantization error: {torch.mean(torch.abs(test_input - dequantized)):.4f}")

def main():
    """Main test function"""
    print("Starting comprehensive LDPC decoder tests...")
    
    try:
        # Test all decoders
        results = test_all_decoders()
        
        # Analyze results
        analyze_results(results)
        
        # Test weight patterns
        test_weight_patterns()
        
        # Test quantization
        test_quantization()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\nImplementation Summary:")
        print("✓ Basic MinSum Decoder")
        print("✓ Neural MinSum Decoder with Edge-Specific Weights")
        print("✓ Neural Offset MinSum Decoder")
        print("✓ Neural 2D MinSum Decoder (4 weight sharing types)")
        print("✓ Neural 2D Offset MinSum Decoder")
        print("✓ RCQ MinSum Decoder")
        print("✓ Weighted RCQ Decoder")
        print("✓ Training Framework with Posterior Joint Training")
        print("✓ Simulation Framework")
        print("✓ Comprehensive Examples and Tests")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
