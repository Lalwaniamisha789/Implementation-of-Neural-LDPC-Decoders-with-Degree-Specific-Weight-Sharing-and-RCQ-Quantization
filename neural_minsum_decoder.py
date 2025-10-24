"""
Neural MinSum (N-NMS) Decoder with Edge-Specific Weights
Implementation based on the paper: arXiv:2310.15483v2

This module implements the original Neural MinSum decoder where each edge
has its own distinct weight parameter for each iteration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from ldpc_decoder import LDPCCode
import logging

logger = logging.getLogger(__name__)

class NeuralMinSumDecoder(nn.Module):
    """
    Neural MinSum Decoder with edge-specific weights
    
    This decoder assigns a distinct weight to each edge in each iteration,
    as described in the original neural MinSum paper. This provides the
    best performance but requires the most parameters.
    """
    
    def __init__(self, code: LDPCCode, max_iterations: int = 50):
        """
        Initialize Neural MinSum decoder
        
        Args:
            code: LDPC code
            max_iterations: Maximum number of decoding iterations
        """
        super().__init__()
        self.code = code
        self.max_iterations = max_iterations
        
        # Create weight tensors for each edge and iteration
        self.beta_weights = nn.ParameterDict()
        
        # Count edges
        num_edges = np.sum(code.H)
        
        # Initialize weights for each edge and iteration
        for t in range(max_iterations):
            for i in range(code.H.shape[0]):
                for j in range(code.H.shape[1]):
                    if code.H[i, j] == 1:
                        key = f"iter_{t}_c{i}_v{j}"
                        # Initialize with small random values around 0.7
                        self.beta_weights[key] = nn.Parameter(torch.randn(1) * 0.1 + 0.7)
        
        logger.info(f"Initialized Neural MinSum decoder with {len(self.beta_weights)} parameters")
        logger.info(f"Total edges: {num_edges}, Parameters per iteration: {num_edges}")
    
    def forward(self, llr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of Neural MinSum decoder
        
        Args:
            llr: Log-likelihood ratios from channel
            
        Returns:
            decoded_bits: Decoded codeword
            posterior: Final posterior probabilities
            iterations: Number of iterations used
        """
        device = llr.device
        n = self.code.n
        H = self.code.H
        
        # Initialize messages
        v2c_messages = torch.zeros(n, H.shape[0], device=device)
        c2v_messages = torch.zeros(H.shape[0], n, device=device)
        
        # Initialize with channel LLRs
        for j in range(n):
            neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
            for i in neighbors:
                v2c_messages[j, i] = llr[j]
        
        for iteration in range(self.max_iterations):
            # Check node update with edge-specific neural weights
            for i in range(H.shape[0]):
                neighbors = torch.where(torch.tensor(H[i, :], device=device) == 1)[0]
                if len(neighbors) == 0:
                    continue
                
                # Get incoming messages
                incoming = v2c_messages[neighbors, i]
                
                # Compute signs and magnitudes
                signs = torch.sign(incoming)
                magnitudes = torch.abs(incoming)
                
                # Find minimum and second minimum
                min_idx = torch.argmin(magnitudes)
                min_val = magnitudes[min_idx]
                
                if len(neighbors) > 1:
                    temp_mags = magnitudes.clone()
                    temp_mags[min_idx] = float('inf')
                    min2_val = torch.min(temp_mags)
                else:
                    min2_val = min_val
                
                # Update C2V messages with edge-specific neural weights
                for j_idx, j in enumerate(neighbors):
                    key = f"iter_{iteration}_c{i}_v{j.item()}"
                    beta = self.beta_weights[key]
                    
                    if j_idx == min_idx:
                        c2v_messages[i, j] = beta * min2_val * torch.prod(signs[torch.arange(len(signs), device=device) != j_idx])
                    else:
                        c2v_messages[i, j] = beta * min_val * torch.prod(signs[torch.arange(len(signs), device=device) != j_idx])
            
            # Variable node update
            for j in range(n):
                neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
                if len(neighbors) == 0:
                    continue
                
                # Update V2C messages
                for i in neighbors:
                    other_neighbors = neighbors[neighbors != i]
                    v2c_messages[j, i] = llr[j] + torch.sum(c2v_messages[other_neighbors, j])
            
            # Check convergence
            posterior = llr.clone()
            for j in range(n):
                neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
                posterior[j] += torch.sum(c2v_messages[neighbors, j])
            
            # Check if all parity checks are satisfied
            decoded_bits = (posterior < 0).int()
            syndrome = torch.matmul(torch.tensor(H, device=device, dtype=torch.float32), decoded_bits.float()) % 2
            
            if torch.sum(syndrome) == 0:
                return decoded_bits, posterior, iteration + 1
        
        # Return final decision
        posterior = llr.clone()
        for j in range(n):
            neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
            posterior[j] += torch.sum(c2v_messages[neighbors, j])
        
        decoded_bits = (posterior < 0).int()
        return decoded_bits, posterior, self.max_iterations

class NeuralOffsetMinSumDecoder(nn.Module):
    """
    Neural Offset MinSum Decoder with edge-specific weights
    
    This decoder uses offset MinSum with edge-specific weights,
    combining the neural approach with offset MinSum algorithm.
    """
    
    def __init__(self, code: LDPCCode, max_iterations: int = 50):
        """
        Initialize Neural Offset MinSum decoder
        
        Args:
            code: LDPC code
            max_iterations: Maximum number of decoding iterations
        """
        super().__init__()
        self.code = code
        self.max_iterations = max_iterations
        
        # Create weight tensors for each edge and iteration
        self.beta_weights = nn.ParameterDict()
        
        # Count edges
        num_edges = np.sum(code.H)
        
        # Initialize weights for each edge and iteration
        for t in range(max_iterations):
            for i in range(code.H.shape[0]):
                for j in range(code.H.shape[1]):
                    if code.H[i, j] == 1:
                        key = f"iter_{t}_c{i}_v{j}"
                        # Initialize with small random values around 0.0 (offset)
                        self.beta_weights[key] = nn.Parameter(torch.randn(1) * 0.1)
        
        logger.info(f"Initialized Neural Offset MinSum decoder with {len(self.beta_weights)} parameters")
        logger.info(f"Total edges: {num_edges}, Parameters per iteration: {num_edges}")
    
    def forward(self, llr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of Neural Offset MinSum decoder
        
        Args:
            llr: Log-likelihood ratios from channel
            
        Returns:
            decoded_bits: Decoded codeword
            posterior: Final posterior probabilities
            iterations: Number of iterations used
        """
        device = llr.device
        n = self.code.n
        H = self.code.H
        
        # Initialize messages
        v2c_messages = torch.zeros(n, H.shape[0], device=device)
        c2v_messages = torch.zeros(H.shape[0], n, device=device)
        
        # Initialize with channel LLRs
        for j in range(n):
            neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
            for i in neighbors:
                v2c_messages[j, i] = llr[j]
        
        for iteration in range(self.max_iterations):
            # Check node update with edge-specific neural weights (Offset MinSum)
            for i in range(H.shape[0]):
                neighbors = torch.where(torch.tensor(H[i, :], device=device) == 1)[0]
                if len(neighbors) == 0:
                    continue
                
                # Get incoming messages
                incoming = v2c_messages[neighbors, i]
                
                # Compute signs and magnitudes
                signs = torch.sign(incoming)
                magnitudes = torch.abs(incoming)
                
                # Find minimum and second minimum
                min_idx = torch.argmin(magnitudes)
                min_val = magnitudes[min_idx]
                
                if len(neighbors) > 1:
                    temp_mags = magnitudes.clone()
                    temp_mags[min_idx] = float('inf')
                    min2_val = torch.min(temp_mags)
                else:
                    min2_val = min_val
                
                # Update C2V messages with edge-specific neural weights (Offset MinSum)
                for j_idx, j in enumerate(neighbors):
                    key = f"iter_{iteration}_c{i}_v{j.item()}"
                    beta = self.beta_weights[key]
                    
                    if j_idx == min_idx:
                        raw_msg = min2_val
                    else:
                        raw_msg = min_val
                    
                    # Apply ReLU and offset
                    offset_msg = F.relu(raw_msg - beta)
                    c2v_messages[i, j] = torch.prod(signs[torch.arange(len(signs), device=device) != j_idx]) * offset_msg
            
            # Variable node update
            for j in range(n):
                neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
                if len(neighbors) == 0:
                    continue
                
                # Update V2C messages
                for i in neighbors:
                    other_neighbors = neighbors[neighbors != i]
                    v2c_messages[j, i] = llr[j] + torch.sum(c2v_messages[other_neighbors, j])
            
            # Check convergence
            posterior = llr.clone()
            for j in range(n):
                neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
                posterior[j] += torch.sum(c2v_messages[neighbors, j])
            
            # Check if all parity checks are satisfied
            decoded_bits = (posterior < 0).int()
            syndrome = torch.matmul(torch.tensor(H, device=device, dtype=torch.float32), decoded_bits.float()) % 2
            
            if torch.sum(syndrome) == 0:
                return decoded_bits, posterior, iteration + 1
        
        # Return final decision
        posterior = llr.clone()
        for j in range(n):
            neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
            posterior[j] += torch.sum(c2v_messages[neighbors, j])
        
        decoded_bits = (posterior < 0).int()
        return decoded_bits, posterior, self.max_iterations

def analyze_weight_patterns(decoder: NeuralMinSumDecoder, code: LDPCCode) -> Dict:
    """
    Analyze weight patterns in the Neural MinSum decoder
    
    Args:
        decoder: Trained Neural MinSum decoder
        code: LDPC code
        
    Returns:
        Analysis results
    """
    analysis = {
        'weight_statistics': {},
        'iteration_patterns': {},
        'node_degree_correlations': {}
    }
    
    # Analyze weights by iteration
    for t in range(decoder.max_iterations):
        iteration_weights = []
        for i in range(code.H.shape[0]):
            for j in range(code.H.shape[1]):
                if code.H[i, j] == 1:
                    key = f"iter_{t}_c{i}_v{j}"
                    if key in decoder.beta_weights:
                        iteration_weights.append(decoder.beta_weights[key].item())
        
        if iteration_weights:
            analysis['iteration_patterns'][t] = {
                'mean': np.mean(iteration_weights),
                'std': np.std(iteration_weights),
                'min': np.min(iteration_weights),
                'max': np.max(iteration_weights)
            }
    
    # Analyze weights by node degrees
    check_degrees = code.check_node_degrees
    variable_degrees = code.variable_node_degrees
    
    for dc in set(check_degrees.values()):
        dc_weights = []
        for i in range(code.H.shape[0]):
            if check_degrees[i] == dc:
                for j in range(code.H.shape[1]):
                    if code.H[i, j] == 1:
                        # Get average weight across iterations
                        weights = []
                        for t in range(decoder.max_iterations):
                            key = f"iter_{t}_c{i}_v{j}"
                            if key in decoder.beta_weights:
                                weights.append(decoder.beta_weights[key].item())
                        if weights:
                            dc_weights.append(np.mean(weights))
        
        if dc_weights:
            analysis['node_degree_correlations'][f'check_degree_{dc}'] = {
                'mean': np.mean(dc_weights),
                'std': np.std(dc_weights),
                'count': len(dc_weights)
            }
    
    return analysis

if __name__ == "__main__":
    from ldpc_decoder import create_test_ldpc_code
    
    # Test Neural MinSum decoder
    print("Testing Neural MinSum Decoder with Edge-Specific Weights")
    print("=" * 60)
    
    code = create_test_ldpc_code()
    print(f"LDPC Code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    print(f"Total edges: {np.sum(code.H)}")
    
    # Test Neural MinSum decoder
    print(f"\nTesting Neural MinSum Decoder:")
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
    
    print("Weight statistics by check node degree:")
    for degree, stats in analysis['node_degree_correlations'].items():
        print(f"  {degree}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")
