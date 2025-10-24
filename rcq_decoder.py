"""
RCQ (Reconstruction-Computation-Quantization) Decoder Implementation
Based on the paper: arXiv:2310.15483v2

This module implements:
- Non-uniform quantization functions
- Reconstruction functions
- RCQ MinSum decoder
- Layered RCQ decoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from ldpc_decoder import LDPCCode
import logging

logger = logging.getLogger(__name__)

class NonUniformQuantizer:
    """
    Non-uniform quantizer with power function thresholds
    
    The quantizer uses thresholds of the form: τ_j = C * (j / (2^bc - 1))^γ
    where C controls the maximum magnitude and γ controls non-uniformity
    """
    
    def __init__(self, bc: int, C: float, gamma: float):
        """
        Initialize quantizer
        
        Args:
            bc: Number of bits for quantization (including sign bit)
            C: Maximum magnitude parameter
            gamma: Non-uniformity parameter (gamma=1 gives uniform quantization)
        """
        self.bc = bc
        self.C = C
        self.gamma = gamma
        
        # Calculate thresholds
        self.thresholds = self._calculate_thresholds()
        
        logger.info(f"Initialized quantizer: bc={bc}, C={C}, gamma={gamma}")
    
    def _calculate_thresholds(self) -> List[float]:
        """Calculate quantization thresholds"""
        thresholds = []
        max_idx = 2**(self.bc - 1) - 1
        
        for j in range(max_idx + 1):
            threshold = self.C * (j / (2**(self.bc - 1) - 1))**self.gamma
            thresholds.append(threshold)
        
        return thresholds
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor
        
        Args:
            x: Input tensor to quantize
            
        Returns:
            Quantized tensor
        """
        device = x.device
        bc = self.bc
        
        # Separate sign and magnitude
        sign = torch.sign(x)
        magnitude = torch.abs(x)
        
        # Quantize magnitude
        quantized_mag = torch.zeros_like(magnitude, dtype=torch.long)
        
        for i, threshold in enumerate(self.thresholds):
            mask = magnitude >= threshold
            quantized_mag[mask] = i
        
        # Handle values above maximum threshold
        max_threshold = self.thresholds[-1]
        quantized_mag[magnitude >= max_threshold] = 2**(bc - 1) - 1
        
        # Combine sign and magnitude
        sign_bit = (sign < 0).long()
        quantized = sign_bit * (2**(bc - 1)) + quantized_mag
        
        return quantized
    
    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Dequantize quantized tensor
        
        Args:
            quantized: Quantized tensor
            
        Returns:
            Dequantized tensor
        """
        device = quantized.device
        bc = self.bc
        
        # Extract sign and magnitude
        sign_bit = (quantized >= 2**(bc - 1)).long()
        magnitude_idx = quantized % (2**(bc - 1))
        
        # Reconstruct magnitude
        reconstructed_mag = torch.zeros_like(quantized, dtype=torch.float32)
        
        for i, threshold in enumerate(self.thresholds):
            mask = magnitude_idx == i
            reconstructed_mag[mask] = threshold
        
        # Apply sign
        sign = 1 - 2 * sign_bit.float()
        reconstructed = sign * reconstructed_mag
        
        return reconstructed

class RCQMinSumDecoder:
    """
    RCQ MinSum Decoder with non-uniform quantization
    """
    
    def __init__(self, code: LDPCCode, bc: int, bv: int, quantizer_params: List[Tuple[float, float]], 
                 max_iterations: int = 50, layered: bool = False):
        """
        Initialize RCQ MinSum decoder
        
        Args:
            code: LDPC code
            bc: Bit width for C2V messages
            bv: Bit width for V2C messages and posteriors
            quantizer_params: List of (C, gamma) parameters for quantizers
            max_iterations: Maximum number of iterations
            layered: Whether to use layered decoding
        """
        self.code = code
        self.bc = bc
        self.bv = bv
        self.max_iterations = max_iterations
        self.layered = layered
        
        # Create quantizers for different iterations/phases
        self.quantizers = []
        for C, gamma in quantizer_params:
            quantizer = NonUniformQuantizer(bc, C, gamma)
            self.quantizers.append(quantizer)
        
        logger.info(f"Initialized RCQ MinSum decoder: bc={bc}, bv={bv}, "
                   f"layered={layered}, {len(self.quantizers)} quantizers")
    
    def _get_quantizer(self, iteration: int) -> NonUniformQuantizer:
        """Get quantizer for given iteration"""
        if len(self.quantizers) == 1:
            return self.quantizers[0]
        
        # Use different quantizers for different phases
        if iteration < self.max_iterations // 3:
            return self.quantizers[0]
        elif iteration < 2 * self.max_iterations // 3:
            return self.quantizers[1] if len(self.quantizers) > 1 else self.quantizers[0]
        else:
            return self.quantizers[-1]
    
    def decode(self, llr: torch.Tensor) -> Tuple[torch.Tensor, bool, int]:
        """
        Decode using RCQ MinSum algorithm
        
        Args:
            llr: Log-likelihood ratios from channel
            
        Returns:
            decoded_bits: Decoded codeword
            success: Whether decoding was successful
            iterations: Number of iterations used
        """
        device = llr.device
        n = self.code.n
        H = self.code.H
        
        if self.layered:
            return self._decode_layered(llr)
        else:
            return self._decode_flooding(llr)
    
    def _decode_flooding(self, llr: torch.Tensor) -> Tuple[torch.Tensor, bool, int]:
        """Flooding schedule RCQ decoding"""
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
            # Get quantizer for this iteration
            quantizer = self._get_quantizer(iteration)
            
            # Check node update
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
                
                # Update C2V messages
                for j_idx, j in enumerate(neighbors):
                    if j_idx == min_idx:
                        raw_msg = min2_val
                    else:
                        raw_msg = min_val
                    
                    # Apply sign
                    signed_msg = torch.prod(signs[torch.arange(len(signs), device=device) != j_idx]) * raw_msg
                    
                    # Quantize and dequantize
                    quantized = quantizer.quantize(signed_msg)
                    c2v_messages[i, j] = quantizer.dequantize(quantized)
            
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
                return decoded_bits, True, iteration + 1
        
        # Return final decision
        posterior = llr.clone()
        for j in range(n):
            neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
            posterior[j] += torch.sum(c2v_messages[neighbors, j])
        
        decoded_bits = (posterior < 0).int()
        return decoded_bits, False, self.max_iterations
    
    def _decode_layered(self, llr: torch.Tensor) -> Tuple[torch.Tensor, bool, int]:
        """Layered schedule RCQ decoding"""
        device = llr.device
        n = self.code.n
        H = self.code.H
        
        # Initialize posteriors
        posteriors = llr.clone()
        
        for iteration in range(self.max_iterations):
            # Get quantizer for this iteration
            quantizer = self._get_quantizer(iteration)
            
            # Process each check node (layer)
            for i in range(H.shape[0]):
                neighbors = torch.where(torch.tensor(H[i, :], device=device) == 1)[0]
                if len(neighbors) == 0:
                    continue
                
                # Subtract previous C2V messages
                for j in neighbors:
                    posteriors[j] -= c2v_messages[i, j] if 'c2v_messages' in locals() else 0
                
                # Compute new C2V messages
                incoming = posteriors[neighbors]
                
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
                
                # Update C2V messages
                c2v_messages = torch.zeros(H.shape[0], n, device=device)
                for j_idx, j in enumerate(neighbors):
                    if j_idx == min_idx:
                        raw_msg = min2_val
                    else:
                        raw_msg = min_val
                    
                    # Apply sign
                    signed_msg = torch.prod(signs[torch.arange(len(signs), device=device) != j_idx]) * raw_msg
                    
                    # Quantize and dequantize
                    quantized = quantizer.quantize(signed_msg)
                    c2v_messages[i, j] = quantizer.dequantize(quantized)
                
                # Add new C2V messages to posteriors
                for j in neighbors:
                    posteriors[j] += c2v_messages[i, j]
            
            # Check convergence
            decoded_bits = (posteriors < 0).int()
            syndrome = torch.matmul(torch.tensor(H, device=device, dtype=torch.float32), decoded_bits.float()) % 2
            
            if torch.sum(syndrome) == 0:
                return decoded_bits, True, iteration + 1
        
        # Return final decision
        decoded_bits = (posteriors < 0).int()
        return decoded_bits, False, self.max_iterations

class WeightedRCQDecoder(nn.Module):
    """
    Weighted RCQ Decoder combining neural weights with RCQ quantization
    """
    
    def __init__(self, code: LDPCCode, bc: int, bv: int, quantizer_params: List[Tuple[float, float]], 
                 weight_sharing_type: int = 2, max_iterations: int = 50, layered: bool = False):
        """
        Initialize Weighted RCQ decoder
        
        Args:
            code: LDPC code
            bc: Bit width for C2V messages
            bv: Bit width for V2C messages and posteriors
            quantizer_params: List of (C, gamma) parameters for quantizers
            weight_sharing_type: Type of weight sharing (1-4)
            max_iterations: Maximum number of iterations
            layered: Whether to use layered decoding
        """
        super().__init__()
        self.code = code
        self.bc = bc
        self.bv = bv
        self.weight_sharing_type = weight_sharing_type
        self.max_iterations = max_iterations
        self.layered = layered
        
        # Create quantizers
        self.quantizers = []
        for C, gamma in quantizer_params:
            quantizer = NonUniformQuantizer(bc, C, gamma)
            self.quantizers.append(quantizer)
        
        # Initialize neural weights
        self.beta_weights = nn.ParameterDict()
        self.alpha_weights = nn.ParameterDict()
        
        # Get unique node degrees
        self.check_node_degrees = list(set(code.check_node_degrees.values()))
        self.variable_node_degrees = list(set(code.variable_node_degrees.values()))
        
        self._initialize_weights()
        
        logger.info(f"Initialized Weighted RCQ decoder: bc={bc}, bv={bv}, "
                   f"weight_type={weight_sharing_type}, layered={layered}")
    
    def _initialize_weights(self):
        """Initialize neural weights based on sharing type"""
        if self.weight_sharing_type == 1:
            # Type 1: Same weight for same (check_degree, variable_degree) pairs
            for t in range(self.max_iterations):
                for dc in self.check_node_degrees:
                    for dv in self.variable_node_degrees:
                        key = f"iter_{t}_dc{dc}_dv{dv}"
                        self.beta_weights[key] = nn.Parameter(torch.randn(1) * 0.1)
        
        elif self.weight_sharing_type == 2:
            # Type 2: Separate weights for check degree and variable degree
            for t in range(self.max_iterations):
                for dc in self.check_node_degrees:
                    key = f"iter_{t}_dc{dc}"
                    self.beta_weights[key] = nn.Parameter(torch.randn(1) * 0.1)
                
                for dv in self.variable_node_degrees:
                    key = f"iter_{t}_dv{dv}"
                    self.alpha_weights[key] = nn.Parameter(torch.randn(1) * 0.1)
        
        elif self.weight_sharing_type == 3:
            # Type 3: Only check node degree based weights
            for t in range(self.max_iterations):
                for dc in self.check_node_degrees:
                    key = f"iter_{t}_dc{dc}"
                    self.beta_weights[key] = nn.Parameter(torch.randn(1) * 0.1)
        
        elif self.weight_sharing_type == 4:
            # Type 4: Only variable node degree based weights
            for t in range(self.max_iterations):
                for dv in self.variable_node_degrees:
                    key = f"iter_{t}_dv{dv}"
                    self.alpha_weights[key] = nn.Parameter(torch.randn(1) * 0.1)
    
    def _get_beta_weight(self, iteration: int, check_node: int, variable_node: int) -> torch.Tensor:
        """Get beta weight based on sharing type"""
        device = next(self.parameters()).device
        
        if self.weight_sharing_type == 1:
            dc = self.code.check_node_degrees[check_node]
            dv = self.code.variable_node_degrees[variable_node]
            key = f"iter_{iteration}_dc{dc}_dv{dv}"
            return self.beta_weights.get(key, torch.tensor(0.7, device=device))
        
        elif self.weight_sharing_type == 2:
            dc = self.code.check_node_degrees[check_node]
            key = f"iter_{iteration}_dc{dc}"
            return self.beta_weights.get(key, torch.tensor(0.7, device=device))
        
        elif self.weight_sharing_type == 3:
            dc = self.code.check_node_degrees[check_node]
            key = f"iter_{iteration}_dc{dc}"
            return self.beta_weights.get(key, torch.tensor(0.7, device=device))
        
        elif self.weight_sharing_type == 4:
            return torch.tensor(0.7, device=device)
        
        else:
            return torch.tensor(0.7, device=device)
    
    def _get_alpha_weight(self, iteration: int, check_node: int, variable_node: int) -> torch.Tensor:
        """Get alpha weight based on sharing type"""
        device = next(self.parameters()).device
        
        if self.weight_sharing_type == 1:
            return torch.tensor(1.0, device=device)
        
        elif self.weight_sharing_type == 2:
            dv = self.code.variable_node_degrees[variable_node]
            key = f"iter_{iteration}_dv{dv}"
            return self.alpha_weights.get(key, torch.tensor(1.0, device=device))
        
        elif self.weight_sharing_type == 3:
            return torch.tensor(1.0, device=device)
        
        elif self.weight_sharing_type == 4:
            dv = self.code.variable_node_degrees[variable_node]
            key = f"iter_{iteration}_dv{dv}"
            return self.alpha_weights.get(key, torch.tensor(1.0, device=device))
        
        else:
            return torch.tensor(1.0, device=device)
    
    def _get_quantizer(self, iteration: int) -> NonUniformQuantizer:
        """Get quantizer for given iteration"""
        if len(self.quantizers) == 1:
            return self.quantizers[0]
        
        # Use different quantizers for different phases
        if iteration < self.max_iterations // 3:
            return self.quantizers[0]
        elif iteration < 2 * self.max_iterations // 3:
            return self.quantizers[1] if len(self.quantizers) > 1 else self.quantizers[0]
        else:
            return self.quantizers[-1]
    
    def forward(self, llr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of Weighted RCQ decoder
        
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
            # Get quantizer for this iteration
            quantizer = self._get_quantizer(iteration)
            
            # Check node update with neural weights and quantization
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
                
                # Update C2V messages with neural weights and quantization
                for j_idx, j in enumerate(neighbors):
                    beta = self._get_beta_weight(iteration, i, j.item())
                    
                    if j_idx == min_idx:
                        raw_msg = min2_val
                    else:
                        raw_msg = min_val
                    
                    # Apply sign and neural weight
                    weighted_msg = beta * torch.prod(signs[torch.arange(len(signs), device=device) != j_idx]) * raw_msg
                    
                    # Quantize and dequantize
                    quantized = quantizer.quantize(weighted_msg)
                    c2v_messages[i, j] = quantizer.dequantize(quantized)
            
            # Variable node update with neural weights
            for j in range(n):
                neighbors = torch.where(torch.tensor(H[:, j], device=device) == 1)[0]
                if len(neighbors) == 0:
                    continue
                
                # Update V2C messages
                for i in neighbors:
                    other_neighbors = neighbors[neighbors != i]
                    alpha = self._get_alpha_weight(iteration, i.item(), j)
                    v2c_messages[j, i] = llr[j] + alpha * torch.sum(c2v_messages[other_neighbors, j])
            
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

if __name__ == "__main__":
    from ldpc_decoder import create_test_ldpc_code
    
    # Test RCQ decoder
    code = create_test_ldpc_code()
    
    # Test quantizer
    print("Testing Non-uniform Quantizer")
    quantizer = NonUniformQuantizer(bc=3, C=5.0, gamma=1.5)
    
    test_input = torch.tensor([-3.2, -1.1, 0.5, 2.8, 4.1])
    quantized = quantizer.quantize(test_input)
    dequantized = quantizer.dequantize(quantized)
    
    print(f"Input: {test_input}")
    print(f"Quantized: {quantized}")
    print(f"Dequantized: {dequantized}")
    
    # Test RCQ MinSum decoder
    print("\nTesting RCQ MinSum Decoder")
    quantizer_params = [(3.0, 1.3), (5.0, 1.3), (7.0, 1.3)]  # Three quantizers
    rcq_decoder = RCQMinSumDecoder(code, bc=3, bv=8, quantizer_params=quantizer_params, max_iterations=10)
    
    llr = torch.randn(code.n) * 2
    decoded, success, iterations = rcq_decoder.decode(llr)
    
    print(f"Decoded: {decoded.numpy()}")
    print(f"Success: {success}, Iterations: {iterations}")
    
    # Test Weighted RCQ decoder
    print("\nTesting Weighted RCQ Decoder")
    wrcq_decoder = WeightedRCQDecoder(code, bc=3, bv=8, quantizer_params=quantizer_params, 
                                   weight_sharing_type=2, max_iterations=10)
    
    llr = torch.randn(code.n) * 2
    decoded, posterior, iterations = wrcq_decoder(llr)
    
    print(f"Decoded: {decoded.numpy()}")
    print(f"Iterations: {iterations}")
    print(f"Number of parameters: {len(wrcq_decoder.beta_weights) + len(wrcq_decoder.alpha_weights)}")