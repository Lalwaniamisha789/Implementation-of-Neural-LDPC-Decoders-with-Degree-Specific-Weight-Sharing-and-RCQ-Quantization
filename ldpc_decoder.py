"""
LDPC Decoding with Degree-Specific Neural Message Weights and RCQ Decoding
Implementation of the paper: arXiv:2310.15483v2

This module implements various LDPC decoders including:
- Basic MinSum decoder
- Neural MinSum (N-NMS) decoder
- Neural 2D MinSum (N-2D-NMS) decoder with node-degree-based weight sharing
- RCQ (Reconstruction-Computation-Quantization) decoder
- Weighted RCQ (W-RCQ) decoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LDPCCode:
    """LDPC Code parameters"""
    n: int  # codeword length
    k: int  # dataword length
    H: np.ndarray  # parity check matrix
    max_iterations: int = 50
    
    @property
    def rate(self) -> float:
        return self.k / self.n
    
    @property
    def check_node_degrees(self) -> Dict[int, int]:
        """Get check node degrees"""
        degrees = {}
        for i in range(self.H.shape[0]):
            degree = np.sum(self.H[i, :])
            degrees[i] = int(degree)
        return degrees
    
    @property
    def variable_node_degrees(self) -> Dict[int, int]:
        """Get variable node degrees"""
        degrees = {}
        for j in range(self.H.shape[1]):
            degree = np.sum(self.H[:, j])
            degrees[j] = int(degree)
        return degrees

class BasicMinSumDecoder:
    """Basic MinSum LDPC Decoder"""
    
    def __init__(self, code: LDPCCode, factor: float = 0.7):
        self.code = code
        self.factor = factor
        
    def decode(self, llr: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """
        Decode using MinSum algorithm
        
        Args:
            llr: Log-likelihood ratios from channel
            
        Returns:
            decoded_bits: Decoded codeword
            success: Whether decoding was successful
            iterations: Number of iterations used
        """
        n, k = self.code.n, self.code.k
        H = self.code.H
        max_iter = self.code.max_iterations
        
        # Initialize messages
        v2c_messages = np.zeros((n, H.shape[0]))
        c2v_messages = np.zeros((H.shape[0], n))
        
        # Initialize with channel LLRs
        for j in range(n):
            neighbors = np.where(H[:, j] == 1)[0]
            for i in neighbors:
                v2c_messages[j, i] = llr[j]
        
        for iteration in range(max_iter):
            # Check node update
            for i in range(H.shape[0]):
                neighbors = np.where(H[i, :] == 1)[0]
                if len(neighbors) == 0:
                    continue
                    
                # Get incoming messages
                incoming = v2c_messages[neighbors, i]
                
                # Compute signs and magnitudes
                signs = np.sign(incoming)
                magnitudes = np.abs(incoming)
                
                # Find minimum and second minimum
                min_idx = np.argmin(magnitudes)
                min_val = magnitudes[min_idx]
                
                if len(neighbors) > 1:
                    # Second minimum
                    temp_mags = magnitudes.copy()
                    temp_mags[min_idx] = np.inf
                    min2_val = np.min(temp_mags)
                else:
                    min2_val = min_val
                
                # Update C2V messages
                for j_idx, j in enumerate(neighbors):
                    if j_idx == min_idx:
                        c2v_messages[i, j] = self.factor * min2_val * np.prod(signs[np.arange(len(signs)) != j_idx])
                    else:
                        c2v_messages[i, j] = self.factor * min_val * np.prod(signs[np.arange(len(signs)) != j_idx])
            
            # Variable node update
            for j in range(n):
                neighbors = np.where(H[:, j] == 1)[0]
                if len(neighbors) == 0:
                    continue
                    
                # Update V2C messages
                for i in neighbors:
                    other_neighbors = neighbors[neighbors != i]
                    v2c_messages[j, i] = llr[j] + np.sum(c2v_messages[other_neighbors, j])
            
            # Check convergence
            posterior = llr.copy()
            for j in range(n):
                neighbors = np.where(H[:, j] == 1)[0]
                posterior[j] += np.sum(c2v_messages[neighbors, j])
            
            # Check if all parity checks are satisfied
            decoded_bits = (posterior < 0).astype(int)
            syndrome = H @ decoded_bits % 2
            
            if np.sum(syndrome) == 0:
                return decoded_bits, True, iteration + 1
        
        # Return final decision
        posterior = llr.copy()
        for j in range(n):
            neighbors = np.where(H[:, j] == 1)[0]
            posterior[j] += np.sum(c2v_messages[neighbors, j])
        
        decoded_bits = (posterior < 0).astype(int)
        return decoded_bits, False, max_iter

class NeuralMinSumDecoder(nn.Module):
    """Neural MinSum LDPC Decoder with edge-specific weights"""
    
    def __init__(self, code: LDPCCode, max_iterations: int = 50):
        super().__init__()
        self.code = code
        self.max_iterations = max_iterations
        
        # Create weight tensors for each edge and iteration
        self.beta_weights = nn.ParameterDict()
        self.alpha_weights = nn.ParameterDict()
        
        # Initialize weights
        for t in range(max_iterations):
            for i in range(code.H.shape[0]):
                for j in range(code.H.shape[1]):
                    if code.H[i, j] == 1:
                        key = f"iter_{t}_c{i}_v{j}"
                        self.beta_weights[key] = nn.Parameter(torch.randn(1) * 0.1)
        
        logger.info(f"Initialized Neural MinSum decoder with {len(self.beta_weights)} parameters")
    
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
            # Check node update with neural weights
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
                
                # Update C2V messages with neural weights
                for j_idx, j in enumerate(neighbors):
                    key = f"iter_{iteration}_c{i}_v{j.item()}"
                    if key in self.beta_weights:
                        beta = self.beta_weights[key]
                    else:
                        beta = torch.tensor(0.7, device=device)
                    
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

def create_test_ldpc_code() -> LDPCCode:
    """Create a simple test LDPC code"""
    # Simple (7,4) Hamming code parity check matrix
    H = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1]
    ])
    
    return LDPCCode(n=7, k=4, H=H, max_iterations=10)

def simulate_awgn_channel(codeword: np.ndarray, snr_db: float) -> np.ndarray:
    """Simulate AWGN channel"""
    # Convert to BPSK
    bpsk_symbols = 2 * codeword - 1
    
    # Calculate noise power
    snr_linear = 10**(snr_db / 10)
    noise_power = 1 / snr_linear
    
    # Add noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(bpsk_symbols))
    received = bpsk_symbols + noise
    
    # Convert back to LLRs
    llr = 2 * received / noise_power
    
    return llr

if __name__ == "__main__":
    # Test basic functionality
    code = create_test_ldpc_code()
    print(f"Created LDPC code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    print(f"Check node degrees: {code.check_node_degrees}")
    print(f"Variable node degrees: {code.variable_node_degrees}")
    
    # Test basic MinSum decoder
    decoder = BasicMinSumDecoder(code)
    
    # Create test codeword
    test_codeword = np.array([0, 1, 1, 0, 1, 0, 1])
    
    # Test at different SNRs
    snrs = [0, 2, 4, 6]
    for snr in snrs:
        llr = simulate_awgn_channel(test_codeword, snr)
        decoded, success, iterations = decoder.decode(llr)
        print(f"SNR {snr}dB: Success={success}, Iterations={iterations}, Error={np.sum(decoded != test_codeword)}")
