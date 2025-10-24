"""
Neural 2D MinSum Decoder with Node-Degree-Based Weight Sharing
Implementation of weight sharing schemes from the paper
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from ldpc_decoder import LDPCCode
import logging

logger = logging.getLogger(__name__)

class Neural2DMinSumDecoder(nn.Module):
    """
    Neural 2D MinSum Decoder with node-degree-based weight sharing
    
    Weight sharing types:
    - Type 1: Same weight for edges with same check node degree AND variable node degree
    - Type 2: Separate weights for check node degree and variable node degree
    - Type 3: Only check node degree based weights
    - Type 4: Only variable node degree based weights
    """
    
    def __init__(self, code: LDPCCode, weight_sharing_type: int = 2, max_iterations: int = 50):
        super().__init__()
        self.code = code
        self.weight_sharing_type = weight_sharing_type
        self.max_iterations = max_iterations
        
        # Get unique node degrees
        self.check_node_degrees = list(set(code.check_node_degrees.values()))
        self.variable_node_degrees = list(set(code.variable_node_degrees.values()))
        
        # Initialize weights based on sharing type
        self.beta_weights = nn.ParameterDict()
        self.alpha_weights = nn.ParameterDict()
        
        self._initialize_weights()
        
        logger.info(f"Initialized N-2D-NMS decoder (Type {weight_sharing_type}) with "
                   f"{len(self.beta_weights)} beta weights and {len(self.alpha_weights)} alpha weights")
    
    def _initialize_weights(self):
        """Initialize weights based on sharing type"""
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
        
        else:
            raise ValueError(f"Invalid weight sharing type: {self.weight_sharing_type}")
    
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
    
    def forward(self, llr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of Neural 2D MinSum decoder
        
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
                    beta = self._get_beta_weight(iteration, i, j.item())
                    
                    if j_idx == min_idx:
                        c2v_messages[i, j] = beta * min2_val * torch.prod(signs[torch.arange(len(signs), device=device) != j_idx])
                    else:
                        c2v_messages[i, j] = beta * min_val * torch.prod(signs[torch.arange(len(signs), device=device) != j_idx])
            
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

class Neural2DOffsetMinSumDecoder(nn.Module):
    """
    Neural 2D Offset MinSum Decoder with node-degree-based weight sharing
    """
    
    def __init__(self, code: LDPCCode, weight_sharing_type: int = 2, max_iterations: int = 50):
        super().__init__()
        self.code = code
        self.weight_sharing_type = weight_sharing_type
        self.max_iterations = max_iterations
        
        # Get unique node degrees
        self.check_node_degrees = list(set(code.check_node_degrees.values()))
        self.variable_node_degrees = list(set(code.variable_node_degrees.values()))
        
        # Initialize weights based on sharing type
        self.beta_weights = nn.ParameterDict()
        self.alpha_weights = nn.ParameterDict()
        
        self._initialize_weights()
        
        logger.info(f"Initialized N-2D-OMS decoder (Type {weight_sharing_type}) with "
                   f"{len(self.beta_weights)} beta weights and {len(self.alpha_weights)} alpha weights")
    
    def _initialize_weights(self):
        """Initialize weights based on sharing type"""
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
        
        else:
            raise ValueError(f"Invalid weight sharing type: {self.weight_sharing_type}")
    
    def _get_beta_weight(self, iteration: int, check_node: int, variable_node: int) -> torch.Tensor:
        """Get beta weight based on sharing type"""
        device = next(self.parameters()).device
        
        if self.weight_sharing_type == 1:
            dc = self.code.check_node_degrees[check_node]
            dv = self.code.variable_node_degrees[variable_node]
            key = f"iter_{iteration}_dc{dc}_dv{dv}"
            return self.beta_weights.get(key, torch.tensor(0.0, device=device))
        
        elif self.weight_sharing_type == 2:
            dc = self.code.check_node_degrees[check_node]
            key = f"iter_{iteration}_dc{dc}"
            return self.beta_weights.get(key, torch.tensor(0.0, device=device))
        
        elif self.weight_sharing_type == 3:
            dc = self.code.check_node_degrees[check_node]
            key = f"iter_{iteration}_dc{dc}"
            return self.beta_weights.get(key, torch.tensor(0.0, device=device))
        
        elif self.weight_sharing_type == 4:
            return torch.tensor(0.0, device=device)
        
        else:
            return torch.tensor(0.0, device=device)
    
    def _get_alpha_weight(self, iteration: int, check_node: int, variable_node: int) -> torch.Tensor:
        """Get alpha weight based on sharing type"""
        device = next(self.parameters()).device
        
        if self.weight_sharing_type == 1:
            return torch.tensor(0.0, device=device)
        
        elif self.weight_sharing_type == 2:
            dv = self.code.variable_node_degrees[variable_node]
            key = f"iter_{iteration}_dv{dv}"
            return self.alpha_weights.get(key, torch.tensor(0.0, device=device))
        
        elif self.weight_sharing_type == 3:
            return torch.tensor(0.0, device=device)
        
        elif self.weight_sharing_type == 4:
            dv = self.code.variable_node_degrees[variable_node]
            key = f"iter_{iteration}_dv{dv}"
            return self.alpha_weights.get(key, torch.tensor(0.0, device=device))
        
        else:
            return torch.tensor(0.0, device=device)
    
    def forward(self, llr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of Neural 2D Offset MinSum decoder
        
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
            # Check node update with neural weights (Offset MinSum)
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
                
                # Update C2V messages with neural weights (Offset MinSum)
                for j_idx, j in enumerate(neighbors):
                    beta = self._get_beta_weight(iteration, i, j.item())
                    alpha = self._get_alpha_weight(iteration, i, j.item())
                    
                    if j_idx == min_idx:
                        raw_msg = min2_val
                    else:
                        raw_msg = min_val
                    
                    # Apply ReLU and offset
                    offset_msg = F.relu(raw_msg - beta) - alpha
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

if __name__ == "__main__":
    from ldpc_decoder import create_test_ldpc_code
    
    # Test Neural 2D MinSum decoder
    code = create_test_ldpc_code()
    
    # Test different weight sharing types
    for weight_type in [1, 2, 3, 4]:
        print(f"\nTesting Neural 2D MinSum decoder (Type {weight_type})")
        decoder = Neural2DMinSumDecoder(code, weight_sharing_type=weight_type, max_iterations=10)
        
        # Test with random LLRs
        llr = torch.randn(code.n) * 2  # Random LLRs
        decoded, posterior, iterations = decoder(llr)
        
        print(f"Decoded: {decoded.numpy()}")
        print(f"Iterations: {iterations}")
        print(f"Number of parameters: {len(decoder.beta_weights) + len(decoder.alpha_weights)}")
    
    # Test Neural 2D Offset MinSum decoder
    print(f"\nTesting Neural 2D Offset MinSum decoder (Type 2)")
    decoder_oms = Neural2DOffsetMinSumDecoder(code, weight_sharing_type=2, max_iterations=10)
    
    llr = torch.randn(code.n) * 2
    decoded, posterior, iterations = decoder_oms(llr)
    
    print(f"Decoded: {decoded.numpy()}")
    print(f"Iterations: {iterations}")
    print(f"Number of parameters: {len(decoder_oms.beta_weights) + len(decoder_oms.alpha_weights)}")
