"""
Example Usage and Test Cases for LDPC Decoding Implementation
Based on the paper: arXiv:2310.15483v2

This module provides comprehensive examples and test cases demonstrating
the usage of all implemented LDPC decoders and training frameworks.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
import time

# Import all modules
from ldpc_decoder import LDPCCode, BasicMinSumDecoder, create_test_ldpc_code, simulate_awgn_channel
from neural_minsum_decoder import NeuralMinSumDecoder, NeuralOffsetMinSumDecoder, analyze_weight_patterns
from neural_2d_decoder import Neural2DMinSumDecoder, Neural2DOffsetMinSumDecoder
from rcq_decoder import RCQMinSumDecoder, WeightedRCQDecoder, NonUniformQuantizer
from training_framework import TrainingConfig, PosteriorJointTrainer, GradientExplosionAnalyzer, create_dvbs2_code
from simulation_framework import SimulationConfig, LDPSimulator, create_test_decoders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_decoder():
    """Example: Basic MinSum decoder usage"""
    print("=" * 60)
    print("EXAMPLE 1: Basic MinSum Decoder")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    print(f"LDPC Code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    print(f"Check node degrees: {code.check_node_degrees}")
    print(f"Variable node degrees: {code.variable_node_degrees}")
    
    # Create decoder
    decoder = BasicMinSumDecoder(code, factor=0.7)
    
    # Test with different SNRs
    test_codeword = np.array([0, 1, 1, 0, 1, 0, 1])
    snrs = [0, 2, 4, 6]
    
    print(f"\nTesting with codeword: {test_codeword}")
    print("SNR (dB) | Success | Iterations | Errors")
    print("-" * 40)
    
    for snr in snrs:
        llr = simulate_awgn_channel(test_codeword, snr)
        decoded, success, iterations = decoder.decode(llr)
        errors = np.sum(decoded != test_codeword)
        print(f"{snr:8.1f} | {str(success):7} | {iterations:10} | {errors:6}")

def example_neural_minsum_decoder():
    """Example: Neural MinSum decoder with edge-specific weights"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Neural MinSum Decoder with Edge-Specific Weights")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Test Neural MinSum decoder
    print(f"Testing Neural MinSum Decoder:")
    decoder = NeuralMinSumDecoder(code, max_iterations=10)
    
    # Count parameters
    num_params = len(decoder.beta_weights)
    print(f"Total parameters: {num_params}")
    print(f"Parameters per iteration: {num_params // decoder.max_iterations}")
    
    # Test with random LLRs
    llr = torch.randn(code.n) * 2
    decoded, posterior, iterations = decoder(llr)
    
    print(f"Decoded: {decoded.numpy()}")
    print(f"Iterations: {iterations}")
    
    # Test Neural Offset MinSum decoder
    print(f"\nTesting Neural Offset MinSum Decoder:")
    decoder_oms = NeuralOffsetMinSumDecoder(code, max_iterations=10)
    
    num_params_oms = len(decoder_oms.beta_weights)
    print(f"Total parameters: {num_params_oms}")
    
    llr = torch.randn(code.n) * 2
    decoded, posterior, iterations = decoder_oms(llr)
    
    print(f"Decoded: {decoded.numpy()}")
    print(f"Iterations: {iterations}")
    
    # Analyze weight patterns
    print(f"\nAnalyzing weight patterns:")
    analysis = analyze_weight_patterns(decoder, code)
    
    print("Weight statistics by iteration:")
    for t, stats in analysis['iteration_patterns'].items():
        print(f"  Iteration {t}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

def example_neural_2d_decoder():
    """Example: Neural 2D MinSum decoder usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Neural 2D MinSum Decoder")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Test different weight sharing types
    weight_types = [1, 2, 3, 4]
    
    print("Weight Sharing Types:")
    print("Type 1: Same weight for same (check_degree, variable_degree) pairs")
    print("Type 2: Separate weights for check degree and variable degree")
    print("Type 3: Only check node degree based weights")
    print("Type 4: Only variable node degree based weights")
    
    print(f"\nTesting Neural 2D MinSum decoders:")
    print("Type | Parameters | Decoded Bits")
    print("-" * 35)
    
    for weight_type in weight_types:
        decoder = Neural2DMinSumDecoder(code, weight_sharing_type=weight_type, max_iterations=10)
        
        # Count parameters
        num_params = len(decoder.beta_weights) + len(decoder.alpha_weights)
        
        # Test with random LLRs
        llr = torch.randn(code.n) * 2
        decoded, posterior, iterations = decoder(llr)
        
        print(f"{weight_type:4} | {num_params:10} | {decoded.numpy()}")

def example_rcq_decoder():
    """Example: RCQ decoder usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: RCQ Decoder")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Test quantizer
    print("Testing Non-uniform Quantizer:")
    quantizer = NonUniformQuantizer(bc=3, C=5.0, gamma=1.5)
    
    test_input = torch.tensor([-3.2, -1.1, 0.5, 2.8, 4.1])
    quantized = quantizer.quantize(test_input)
    dequantized = quantizer.dequantize(quantized)
    
    print(f"Input:      {test_input}")
    print(f"Quantized:  {quantized}")
    print(f"Dequantized: {dequantized}")
    
    # Test RCQ MinSum decoder
    print(f"\nTesting RCQ MinSum Decoder:")
    quantizer_params = [(3.0, 1.3), (5.0, 1.3), (7.0, 1.3)]
    rcq_decoder = RCQMinSumDecoder(code, bc=3, bv=8, quantizer_params=quantizer_params, max_iterations=10)
    
    llr = torch.randn(code.n) * 2
    decoded, success, iterations = rcq_decoder.decode(llr)
    
    print(f"Decoded: {decoded.numpy()}")
    print(f"Success: {success}, Iterations: {iterations}")

def example_weighted_rcq_decoder():
    """Example: Weighted RCQ decoder usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Weighted RCQ Decoder")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Test Weighted RCQ decoder
    quantizer_params = [(3.0, 1.3), (5.0, 1.3), (7.0, 1.3)]
    wrcq_decoder = WeightedRCQDecoder(
        code, bc=3, bv=8, quantizer_params=quantizer_params, 
        weight_sharing_type=2, max_iterations=10
    )
    
    print(f"Weighted RCQ Decoder Parameters:")
    print(f"Beta weights: {len(wrcq_decoder.beta_weights)}")
    print(f"Alpha weights: {len(wrcq_decoder.alpha_weights)}")
    print(f"Total parameters: {len(wrcq_decoder.beta_weights) + len(wrcq_decoder.alpha_weights)}")
    
    # Test with random LLRs
    llr = torch.randn(code.n) * 2
    decoded, posterior, iterations = wrcq_decoder(llr)
    
    print(f"\nDecoded: {decoded.numpy()}")
    print(f"Iterations: {iterations}")

def example_training_framework():
    """Example: Training framework usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Training Framework")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Create model
    model = Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=5)
    
    # Training configuration
    config = TrainingConfig(
        batch_size=8,
        num_epochs=5,
        learning_rate=0.001,
        snr_range=(0.0, 3.0),
        snr_step=1.0,
        use_posterior_training=True,
        device='cpu'
    )
    
    print(f"Training Configuration:")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"SNR range: {config.snr_range}")
    
    # Create trainer
    trainer = PosteriorJointTrainer(model, config)
    
    # Train model
    print(f"\nStarting training...")
    history = trainer.train(code, num_train_samples=100, num_val_samples=20)
    
    print(f"\nTraining completed!")
    print(f"Final training accuracy: {history['train_accuracies'][-1]:.4f}")
    print(f"Final training loss: {history['train_losses'][-1]:.6f}")

def example_simulation_framework():
    """Example: Simulation framework usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Simulation Framework")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Simulation configuration
    config = SimulationConfig(
        snr_range=(0.0, 3.0),
        snr_step=1.0,
        max_frames=100,
        max_errors=20,
        parallel_workers=1,
        device='cpu'
    )
    
    print(f"Simulation Configuration:")
    print(f"SNR range: {config.snr_range}")
    print(f"SNR step: {config.snr_step}")
    print(f"Max frames: {config.max_frames}")
    print(f"Max errors: {config.max_errors}")
    
    # Create simulator
    simulator = LDPSimulator(config)
    
    # Create test decoders
    decoders = {
        'Basic MinSum': BasicMinSumDecoder(code, factor=0.7),
        'N-2D-NMS Type 2': Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=5)
    }
    
    print(f"\nTesting {len(decoders)} decoders...")
    
    # Run simulation
    results = simulator.simulate_multiple_decoders(decoders, code)
    
    # Print results summary
    print(f"\nSimulation Results Summary:")
    for name, result in results.items():
        print(f"\n{name}:")
        for i, snr in enumerate(result.snr_values):
            print(f"  SNR {snr:.1f}dB: FER={result.frame_error_rates[i]:.2e}, "
                  f"BER={result.bit_error_rates[i]:.2e}, "
                  f"Avg Iter={result.average_iterations[i]:.1f}")

def example_gradient_explosion_analysis():
    """Example: Gradient explosion analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Gradient Explosion Analysis")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Create model
    model = Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=5)
    
    # Analyze gradient explosion
    analyzer = GradientExplosionAnalyzer(model, code)
    results = analyzer.analyze_gradient_explosion(num_samples=20)
    
    print(f"Gradient Explosion Analysis Results:")
    print(f"Mean gradient magnitude: {results['mean_gradient']:.6f}")
    print(f"Max gradient magnitude: {results['max_gradient']:.6f}")
    print(f"Std gradient magnitude: {results['std_gradient']:.6f}")
    
    # Check for gradient explosion
    if results['max_gradient'] > 10.0:
        print("Gradient explosion detected!")
    else:
        print("No gradient explosion detected")

def example_performance_comparison():
    """Example: Performance comparison of different decoders"""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Performance Comparison")
    print("=" * 60)
    
    # Create test LDPC code
    code = create_test_ldpc_code()
    
    # Create decoders
    decoders = {
        'Basic MinSum': BasicMinSumDecoder(code, factor=0.7),
        'N-2D-NMS Type 1': Neural2DMinSumDecoder(code, weight_sharing_type=1, max_iterations=5),
        'N-2D-NMS Type 2': Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=5),
        'N-2D-OMS Type 2': Neural2DOffsetMinSumDecoder(code, weight_sharing_type=2, max_iterations=5)
    }
    
    # Test at specific SNR
    snr_db = 2.0
    test_codeword = np.zeros(code.n, dtype=int)
    
    print(f"Performance comparison at SNR = {snr_db} dB:")
    print("Decoder Name        | Success | Iterations | Parameters")
    print("-" * 55)
    
    for name, decoder in decoders.items():
        # Generate test data
        llr = simulate_awgn_channel(test_codeword, snr_db)
        
        # Decode
        if isinstance(decoder, torch.nn.Module):
            llr_tensor = torch.tensor(llr, dtype=torch.float32)
            decoded, posterior, iterations = decoder(llr_tensor)
            success = torch.all(decoded == torch.tensor(test_codeword))
            num_params = sum(p.numel() for p in decoder.parameters())
        else:
            decoded, success, iterations = decoder.decode(llr)
            num_params = 1  # Basic decoder has 1 parameter (factor)
        
        print(f"{name:19} | {str(success):7} | {iterations:10} | {num_params:10}")

def example_large_code_simulation():
    """Example: Simulation with larger LDPC code"""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Large Code Simulation")
    print("=" * 60)
    
    # Create larger LDPC code
    code = create_dvbs2_code()
    print(f"Large LDPC Code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    
    # Create decoders
    decoders = {
        'Basic MinSum': BasicMinSumDecoder(code, factor=0.7),
        'N-2D-NMS Type 2': Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=10)
    }
    
    # Simulation configuration for large code
    config = SimulationConfig(
        snr_range=(1.0, 3.0),
        snr_step=1.0,
        max_frames=50,  # Reduced for faster execution
        max_errors=10,
        parallel_workers=1,
        device='cpu'
    )
    
    print(f"Simulation configuration:")
    print(f"SNR range: {config.snr_range}")
    print(f"Max frames per SNR: {config.max_frames}")
    
    # Create simulator
    simulator = LDPSimulator(config)
    
    # Run simulation
    print(f"\nRunning simulation...")
    start_time = time.time()
    results = simulator.simulate_multiple_decoders(decoders, code)
    simulation_time = time.time() - start_time
    
    print(f"Simulation completed in {simulation_time:.2f} seconds")
    
    # Print results
    for name, result in results.items():
        print(f"\n{name} Results:")
        for i, snr in enumerate(result.snr_values):
            print(f"  SNR {snr:.1f}dB: FER={result.frame_error_rates[i]:.2e}, "
                  f"BER={result.bit_error_rates[i]:.2e}")

def run_all_examples():
    """Run all examples"""
    print("LDPC Decoding Implementation Examples")
    print("Based on: arXiv:2310.15483v2")
    print("=" * 80)
    
    try:
        example_basic_decoder()
        example_neural_minsum_decoder()
        example_neural_2d_decoder()
        example_rcq_decoder()
        example_weighted_rcq_decoder()
        example_training_framework()
        example_simulation_framework()
        example_gradient_explosion_analysis()
        example_performance_comparison()
        example_large_code_simulation()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        logger.error(f"Error running examples: {e}")

def run_quick_test():
    """Run a quick test of basic functionality"""
    print("Quick Test of LDPC Decoding Implementation")
    print("=" * 50)
    
    try:
        # Test basic decoder
        code = create_test_ldpc_code()
        decoder = BasicMinSumDecoder(code, factor=0.7)
        
        test_codeword = np.zeros(code.n, dtype=int)
        llr = simulate_awgn_channel(test_codeword, 2.0)
        decoded, success, iterations = decoder.decode(llr)
        
        print(f"Basic MinSum Test:")
        print(f"  Code: ({code.n}, {code.k})")
        print(f"  Success: {success}")
        print(f"  Iterations: {iterations}")
        print(f"  Errors: {np.sum(decoded != test_codeword)}")
        
        # Test neural decoder
        neural_decoder = NeuralMinSumDecoder(code, max_iterations=5)
        llr_tensor = torch.tensor(llr, dtype=torch.float32)
        decoded_neural, posterior, iterations_neural = neural_decoder(llr_tensor)
        
        print(f"\nNeural MinSum Test:")
        print(f"  Success: {torch.all(decoded_neural == torch.tensor(test_codeword))}")
        print(f"  Iterations: {iterations_neural}")
        print(f"  Parameters: {sum(p.numel() for p in neural_decoder.parameters())}")
        
        print(f"\n[SUCCESS] Quick test passed!")
        
    except Exception as e:
        print(f"[ERROR] Quick test failed: {e}")
        logger.error(f"Quick test failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_test()
    else:
        run_all_examples()
