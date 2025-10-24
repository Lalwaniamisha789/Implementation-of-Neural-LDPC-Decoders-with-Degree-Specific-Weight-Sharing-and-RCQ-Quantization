"""
Training Framework for Neural LDPC Decoders
Implementation of posterior joint training to address gradient explosion
Based on the paper: arXiv:2310.15483v2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from ldpc_decoder import LDPCCode, simulate_awgn_channel
from neural_2d_decoder import Neural2DMinSumDecoder, Neural2DOffsetMinSumDecoder
from rcq_decoder import WeightedRCQDecoder
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    snr_range: Tuple[float, float] = (0.0, 6.0)
    snr_step: float = 0.5
    max_grad_norm: float = 1.0
    use_posterior_training: bool = True
    use_gradient_clipping: bool = False
    clip_threshold: float = 1e-3
    device: str = 'cpu'

class PosteriorJointTrainer:
    """
    Trainer implementing posterior joint training to address gradient explosion
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.gradient_norms = []
        
        logger.info(f"Initialized trainer with {sum(p.numel() for p in model.parameters())} parameters")
    
    def generate_training_data(self, code: LDPCCode, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data with all-zero codewords corrupted by AWGN
        
        Args:
            code: LDPC code
            num_samples: Number of training samples
            
        Returns:
            llrs: Log-likelihood ratios
            targets: Target codewords (all zeros)
        """
        # Generate all-zero codewords
        codewords = torch.zeros(num_samples, code.n, dtype=torch.float32)
        
        # Generate SNRs
        snr_min, snr_max = self.config.snr_range
        snrs = torch.linspace(snr_min, snr_max, num_samples)
        
        # Generate LLRs
        llrs = torch.zeros_like(codewords)
        
        for i in range(num_samples):
            snr_db = snrs[i].item()
            llr = simulate_awgn_channel(codewords[i].numpy(), snr_db)
            llrs[i] = torch.tensor(llr, dtype=torch.float32)
        
        return llrs, codewords
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                    posteriors: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-loss cross entropy
        
        Args:
            outputs: Decoded bits
            targets: Target codewords
            posteriors: Posterior probabilities
            
        Returns:
            Loss value
        """
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(-posteriors, targets.float())
        
        # Additional loss terms can be added here
        return bce_loss
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_grad_norms = []
        
        for batch_idx, (llrs, targets) in enumerate(train_loader):
            llrs = llrs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            decoded, posteriors, iterations = self.model(llrs)
            
            # Compute loss
            loss = self.compute_loss(decoded, targets, posteriors)
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norms
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            epoch_grad_norms.append(total_norm)
            
            # Apply gradient clipping if enabled
            if self.config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_threshold)
            
            # Update parameters
            self.optimizer.step()
            
            # Compute accuracy
            correct = (decoded == targets).all(dim=1).sum().item()
            total_correct += correct
            total_samples += llrs.size(0)
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}, '
                           f'Grad Norm: {total_norm:.6f}, Acc: {correct/llrs.size(0):.4f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / total_samples
        avg_grad_norm = np.mean(epoch_grad_norms)
        
        return avg_loss, avg_accuracy, avg_grad_norm
    
    def train(self, code: LDPCCode, num_train_samples: int = 1000, 
              num_val_samples: int = 200) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            code: LDPC code
            num_train_samples: Number of training samples
            num_val_samples: Number of validation samples
            
        Returns:
            Training history
        """
        logger.info("Generating training data...")
        
        # Generate training data
        train_llrs, train_targets = self.generate_training_data(code, num_train_samples)
        val_llrs, val_targets = self.generate_training_data(code, num_val_samples)
        
        # Create data loaders
        train_dataset = TensorDataset(train_llrs, train_targets)
        val_dataset = TensorDataset(val_llrs, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc, train_grad_norm = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_grad_norm = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.gradient_norms.append(train_grad_norm)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{self.config.num_epochs}: '
                        f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, '
                        f'Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, '
                        f'Grad Norm: {train_grad_norm:.6f}, Time: {epoch_time:.2f}s')
            
            # Early stopping could be added here
            if train_acc > 0.99:  # Stop if accuracy is very high
                logger.info(f"Early stopping at epoch {epoch+1} due to high accuracy")
                break
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'gradient_norms': self.gradient_norms
        }
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss, accuracy, and gradient norm
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for llrs, targets in val_loader:
                llrs = llrs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                decoded, posteriors, iterations = self.model(llrs)
                
                # Compute loss
                loss = self.compute_loss(decoded, targets, posteriors)
                
                # Compute accuracy
                correct = (decoded == targets).all(dim=1).sum().item()
                total_correct += correct
                total_samples += llrs.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy, 0.0  # No gradient norm for validation
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.train_losses)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.train_accuracies)
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True)
        
        # Gradient norms
        axes[2].plot(self.gradient_norms)
        axes[2].set_title('Gradient Norms')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Gradient Norm')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

class GradientExplosionAnalyzer:
    """
    Analyzer for gradient explosion issues in neural LDPC decoders
    """
    
    def __init__(self, model: nn.Module, code: LDPCCode):
        self.model = model
        self.code = code
    
    def analyze_gradient_explosion(self, num_samples: int = 100) -> Dict[str, List[float]]:
        """
        Analyze gradient explosion by computing gradients for random inputs
        
        Args:
            num_samples: Number of samples to analyze
            
        Returns:
            Analysis results
        """
        self.model.eval()
        
        gradient_magnitudes = []
        iteration_gradients = []
        
        for sample_idx in range(num_samples):
            # Generate random input
            llr = torch.randn(self.code.n) * 2
            
            # Forward pass
            decoded, posteriors, iterations = self.model(llr)
            
            # Compute gradients
            loss = F.binary_cross_entropy_with_logits(-posteriors, torch.zeros_like(decoded).float())
            loss.backward()
            
            # Compute gradient magnitudes
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            gradient_magnitudes.append(total_norm)
            iteration_gradients.append(iterations)
            
            # Clear gradients
            self.model.zero_grad()
        
        return {
            'gradient_magnitudes': gradient_magnitudes,
            'iteration_counts': iteration_gradients,
            'mean_gradient': np.mean(gradient_magnitudes),
            'std_gradient': np.std(gradient_magnitudes),
            'max_gradient': np.max(gradient_magnitudes)
        }
    
    def plot_gradient_analysis(self, results: Dict[str, List[float]], 
                               save_path: Optional[str] = None):
        """Plot gradient analysis results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gradient magnitudes histogram
        axes[0].hist(results['gradient_magnitudes'], bins=20, alpha=0.7)
        axes[0].set_title('Gradient Magnitude Distribution')
        axes[0].set_xlabel('Gradient Magnitude')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)
        
        # Gradient vs iterations
        axes[1].scatter(results['iteration_counts'], results['gradient_magnitudes'], alpha=0.6)
        axes[1].set_title('Gradient Magnitude vs Iterations')
        axes[1].set_xlabel('Iterations')
        axes[1].set_ylabel('Gradient Magnitude')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def create_dvbs2_code() -> LDPCCode:
    """Create a DVBS-2 LDPC code for testing"""
    # Simplified DVBS-2 code structure
    # In practice, you would load the actual parity check matrix
    n, k = 16200, 7200
    
    # Create a random parity check matrix for demonstration
    # In practice, use the actual DVBS-2 matrix
    np.random.seed(42)
    H = np.random.randint(0, 2, (n-k, n))
    
    # Ensure each row has some ones (check nodes have degree > 0)
    for i in range(n-k):
        if np.sum(H[i, :]) == 0:
            H[i, np.random.randint(0, n)] = 1
    
    # Ensure each column has some ones (variable nodes have degree > 0)
    for j in range(n):
        if np.sum(H[:, j]) == 0:
            H[np.random.randint(0, n-k), j] = 1
    
    return LDPCCode(n=n, k=k, H=H, max_iterations=50)

if __name__ == "__main__":
    # Test training framework
    print("Testing Neural LDPC Decoder Training Framework")
    
    # Create test code
    code = create_dvbs2_code()
    print(f"Created LDPC code: ({code.n}, {code.k}) with rate {code.rate:.3f}")
    
    # Create model
    model = Neural2DMinSumDecoder(code, weight_sharing_type=2, max_iterations=10)
    
    # Training configuration
    config = TrainingConfig(
        batch_size=16,
        num_epochs=20,
        learning_rate=0.001,
        snr_range=(0.0, 4.0),
        snr_step=0.5,
        use_posterior_training=True,
        device='cpu'
    )
    
    # Create trainer
    trainer = PosteriorJointTrainer(model, config)
    
    # Train model
    print("Starting training...")
    history = trainer.train(code, num_train_samples=500, num_val_samples=100)
    
    # Plot results
    trainer.plot_training_history()
    
    # Analyze gradient explosion
    print("\nAnalyzing gradient explosion...")
    analyzer = GradientExplosionAnalyzer(model, code)
    grad_results = analyzer.analyze_gradient_explosion(num_samples=50)
    
    print(f"Mean gradient magnitude: {grad_results['mean_gradient']:.6f}")
    print(f"Max gradient magnitude: {grad_results['max_gradient']:.6f}")
    
    analyzer.plot_gradient_analysis(grad_results)
    
    print("Training completed!")