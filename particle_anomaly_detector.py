"""
Deep Autoencoder for Particle Collision Anomaly Detection

This module implements a deep autoencoder to detect anomalous events from particle 
collision datasets in a model-independent way. The goal is to find events that 
significantly deviate from the standard background, potentially indicating new 
physical phenomena beyond the Standard Model.

Author: STEMENOF Project Gaia
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tqdm import tqdm
import h5py
import os
from typing import Tuple, Optional, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParticleDataPreprocessor:
    """
    Preprocessor for particle collision data featuring:
    - Particle four-momentum (px, py, pz, E)
    - Particle type/charge (identifier)
    - Event-level variables (missing transverse energy, total energy, jet multiplicity)
    """
    
    def __init__(self, max_particles_per_event: int = 100):
        self.max_particles_per_event = max_particles_per_event
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.is_fitted = False
        
    def extract_features(self, event_data: Dict) -> np.ndarray:
        """
        Extract and engineer features from particle collision event data.
        
        Args:
            event_data: Dictionary containing particle and event-level information
            
        Returns:
            Flattened feature vector for the event
        """
        features = []
        
        # Particle-level features
        particles = event_data.get('particles', [])
        n_particles = min(len(particles), self.max_particles_per_event)
        
        # Four-momentum features (px, py, pz, E) for each particle
        particle_features = np.zeros((self.max_particles_per_event, 5))  # 4-momentum + type
        
        for i in range(n_particles):
            particle = particles[i]
            particle_features[i, 0] = particle.get('px', 0)
            particle_features[i, 1] = particle.get('py', 0) 
            particle_features[i, 2] = particle.get('pz', 0)
            particle_features[i, 3] = particle.get('E', 0)
            particle_features[i, 4] = particle.get('type', 0)  # Particle type/charge
            
        # Flatten particle features
        features.extend(particle_features.flatten())
        
        # Event-level features
        event_features = [
            event_data.get('missing_et', 0),      # Missing transverse energy
            event_data.get('total_energy', 0),    # Total energy
            event_data.get('jet_multiplicity', 0), # Number of jets
            event_data.get('lepton_multiplicity', 0), # Number of leptons
            event_data.get('n_particles', len(particles)), # Actual number of particles
        ]
        
        # High-level derived features
        # Scalar sum of transverse momentum
        pt_sum = sum([np.sqrt(p.get('px', 0)**2 + p.get('py', 0)**2) 
                     for p in particles[:n_particles]])
        event_features.append(pt_sum)
        
        # Sphericity and other event shape variables
        if n_particles > 0:
            # Simplified sphericity calculation
            px_vals = [p.get('px', 0) for p in particles[:n_particles]]
            py_vals = [p.get('py', 0) for p in particles[:n_particles]]
            pz_vals = [p.get('pz', 0) for p in particles[:n_particles]]
            
            sphericity = self._calculate_sphericity(px_vals, py_vals, pz_vals)
            event_features.append(sphericity)
            
            # Aplanarity 
            aplanarity = self._calculate_aplanarity(px_vals, py_vals, pz_vals)
            event_features.append(aplanarity)
        else:
            event_features.extend([0, 0])  # sphericity, aplanarity
            
        features.extend(event_features)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_sphericity(self, px_vals: List[float], py_vals: List[float], 
                            pz_vals: List[float]) -> float:
        """Calculate sphericity tensor eigenvalues for event shape analysis"""
        if not px_vals:
            return 0.0
            
        # Momentum tensor
        p_total = sum([px**2 + py**2 + pz**2 for px, py, pz in zip(px_vals, py_vals, pz_vals)])
        if p_total == 0:
            return 0.0
            
        # Simplified sphericity approximation
        pt_sum = sum([np.sqrt(px**2 + py**2) for px, py in zip(px_vals, py_vals)])
        return pt_sum / np.sqrt(p_total) if p_total > 0 else 0.0
    
    def _calculate_aplanarity(self, px_vals: List[float], py_vals: List[float], 
                            pz_vals: List[float]) -> float:
        """Calculate aplanarity for event shape analysis"""
        if len(px_vals) < 3:
            return 0.0
        
        # Simplified aplanarity approximation    
        pz_sum = sum([abs(pz) for pz in pz_vals])
        p_total = sum([np.sqrt(px**2 + py**2 + pz**2) for px, py, pz in zip(px_vals, py_vals, pz_vals)])
        return pz_sum / p_total if p_total > 0 else 0.0
    
    def fit_transform(self, events_data: List[Dict]) -> np.ndarray:
        """Fit preprocessor and transform events data"""
        features_matrix = []
        
        for event in tqdm(events_data, desc="Processing events"):
            features = self.extract_features(event)
            features_matrix.append(features)
            
        features_matrix = np.array(features_matrix)
        
        # Handle NaN and infinite values
        features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit and transform with robust scaling
        features_matrix = self.scaler.fit_transform(features_matrix)
        self.is_fitted = True
        
        return features_matrix
    
    def transform(self, events_data: List[Dict]) -> np.ndarray:
        """Transform events data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        features_matrix = []
        
        for event in events_data:
            features = self.extract_features(event)
            features_matrix.append(features)
            
        features_matrix = np.array(features_matrix)
        features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.scaler.transform(features_matrix)


class DeepAutoencoder(nn.Module):
    """
    Deep Autoencoder for particle collision anomaly detection.
    
    Architecture:
    - Encoder: Progressive dimensionality reduction with skip connections
    - Latent space: Compressed representation 
    - Decoder: Reconstruction with symmetric architecture
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: List[int] = None):
        super(DeepAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
            
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (symmetric to encoder)
        decoder_layers = [nn.Linear(latent_dim, hidden_dims[-1])]
        prev_dim = hidden_dims[-1]
        
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.extend([
            nn.ReLU(),
            nn.Linear(prev_dim, input_dim)
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent representation"""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class ParticleAnomalyDetector:
    """
    Complete pipeline for particle collision anomaly detection using deep autoencoders.
    """
    
    def __init__(self, max_particles_per_event: int = 100, latent_dim: int = 32, 
                 hidden_dims: List[int] = None, device: str = None):
        self.max_particles_per_event = max_particles_per_event
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256, 128, 64]
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = ParticleDataPreprocessor(max_particles_per_event)
        self.model = None
        self.is_trained = False
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, events_data: List[Dict], validation_split: float = 0.2) -> Tuple:
        """Prepare and split data for training"""
        logger.info(f"Preparing {len(events_data)} events...")
        
        # Extract features
        features = self.preprocessor.fit_transform(events_data)
        
        # Split data
        train_features, val_features = train_test_split(
            features, test_size=validation_split, random_state=42
        )
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_features).to(self.device)
        val_tensor = torch.FloatTensor(val_features).to(self.device)
        
        logger.info(f"Training set: {train_tensor.shape}, Validation set: {val_tensor.shape}")
        
        return train_tensor, val_tensor
    
    def build_model(self, input_dim: int):
        """Build the autoencoder model"""
        self.model = DeepAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        logger.info(f"Model architecture: {self.model}")
        
    def train(self, train_data: torch.Tensor, val_data: torch.Tensor, 
              epochs: int = 100, batch_size: int = 256, learning_rate: float = 1e-3,
              early_stopping_patience: int = 10):
        """Train the autoencoder"""
        
        if self.model is None:
            self.build_model(train_data.shape[1])
            
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        # Data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_data)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_data, _ in train_loader:
                optimizer.zero_grad()
                
                reconstruction, latent = self.model(batch_data)
                loss = criterion(reconstruction, batch_data)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    reconstruction, latent = self.model(batch_data)
                    loss = criterion(reconstruction, batch_data)
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_autoencoder.pth')
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: Train Loss = {train_loss:.6f}, "
                          f"Val Loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        # Load best model
        self.model.load_state_dict(torch.load('best_autoencoder.pth'))
        self.is_trained = True
        logger.info("Training completed!")
        
    def compute_anomaly_scores(self, events_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for events based on reconstruction error and latent space distance.
        
        Returns:
            reconstruction_errors: MSE between input and reconstruction
            latent_distances: Distance from centroid in latent space
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing anomaly scores")
            
        # Preprocess data
        features = self.preprocessor.transform(events_data)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        self.model.eval()
        reconstruction_errors = []
        latent_representations = []
        
        with torch.no_grad():
            # Process in batches to handle memory constraints
            batch_size = 1000
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i+batch_size]
                reconstruction, latent = self.model(batch)
                
                # Reconstruction error (MSE per sample)
                mse = torch.mean((batch - reconstruction) ** 2, dim=1)
                reconstruction_errors.extend(mse.cpu().numpy())
                
                # Store latent representations
                latent_representations.extend(latent.cpu().numpy())
                
        reconstruction_errors = np.array(reconstruction_errors)
        latent_representations = np.array(latent_representations)
        
        # Compute latent space distances from centroid
        latent_centroid = np.mean(latent_representations, axis=0)
        latent_distances = np.linalg.norm(latent_representations - latent_centroid, axis=1)
        
        return reconstruction_errors, latent_distances
    
    def detect_anomalies(self, events_data: List[Dict], 
                        contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Detect anomalous events using combined scoring approach.
        
        Args:
            events_data: List of event dictionaries
            contamination: Expected fraction of anomalous events
            
        Returns:
            anomaly_labels: Binary labels (1 for anomaly, 0 for normal)
            anomaly_info: Dictionary with scores and thresholds
        """
        # Compute anomaly scores
        recon_errors, latent_distances = self.compute_anomaly_scores(events_data)
        
        # Normalize scores to [0, 1]
        recon_errors_norm = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min())
        latent_distances_norm = (latent_distances - latent_distances.min()) / (latent_distances.max() - latent_distances.min())
        
        # Combined anomaly score (weighted average)
        combined_scores = 0.7 * recon_errors_norm + 0.3 * latent_distances_norm
        
        # Determine threshold based on contamination
        threshold = np.percentile(combined_scores, (1 - contamination) * 100)
        
        # Binary anomaly labels
        anomaly_labels = (combined_scores > threshold).astype(int)
        
        anomaly_info = {
            'reconstruction_errors': recon_errors,
            'latent_distances': latent_distances,
            'combined_scores': combined_scores,
            'threshold': threshold,
            'n_anomalies': np.sum(anomaly_labels)
        }
        
        logger.info(f"Detected {anomaly_info['n_anomalies']} anomalous events "
                   f"({anomaly_info['n_anomalies']/len(events_data)*100:.2f}%)")
        
        return anomaly_labels, anomaly_info
    
    def plot_training_history(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Autoencoder Training History')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_anomaly_analysis(self, anomaly_info: Dict, labels: Optional[np.ndarray] = None):
        """
        Plot comprehensive anomaly analysis including score distributions and ROC curve.
        
        Args:
            anomaly_info: Dictionary from detect_anomalies method
            labels: True labels if available for evaluation
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reconstruction error distribution
        axes[0, 0].hist(anomaly_info['reconstruction_errors'], bins=50, alpha=0.7, density=True)
        axes[0, 0].set_xlabel('Reconstruction Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Reconstruction Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Latent distance distribution  
        axes[0, 1].hist(anomaly_info['latent_distances'], bins=50, alpha=0.7, density=True)
        axes[0, 1].set_xlabel('Latent Space Distance')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Latent Distance Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Combined score distribution with threshold
        axes[1, 0].hist(anomaly_info['combined_scores'], bins=50, alpha=0.7, density=True)
        axes[1, 0].axvline(anomaly_info['threshold'], color='red', linestyle='--', 
                          label=f"Threshold: {anomaly_info['threshold']:.3f}")
        axes[1, 0].set_xlabel('Combined Anomaly Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Combined Anomaly Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ROC curve (if true labels available)
        if labels is not None:
            fpr, tpr, _ = roc_curve(labels, anomaly_info['combined_scores'])
            auc_score = roc_auc_score(labels, anomaly_info['combined_scores'])
            
            axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('ROC Curve')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'ROC Curve\n(requires true labels)', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_xlim([0, 1])
            axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath: str):
        """Save the trained model and preprocessor"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'preprocessor': self.preprocessor,
            'config': {
                'max_particles_per_event': self.max_particles_per_event,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims,
                'input_dim': self.model.input_dim
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load a trained model and preprocessor"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load configuration
        config = checkpoint['config']
        self.max_particles_per_event = config['max_particles_per_event']
        self.latent_dim = config['latent_dim']
        self.hidden_dims = config['hidden_dims']
        
        # Rebuild model with correct dimensions
        self.build_model(config['input_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load preprocessor
        self.preprocessor = checkpoint['preprocessor']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


def generate_sample_collider_data(n_events: int = 10000, anomaly_fraction: float = 0.05) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate synthetic particle collision data mimicking LHC-style events.
    
    This creates simulated background events and injects anomalous events that could 
    represent new physics beyond the Standard Model.
    
    Args:
        n_events: Total number of events to generate
        anomaly_fraction: Fraction of events that are anomalous
        
    Returns:
        events_data: List of event dictionaries
        true_labels: Binary labels (1 for anomaly, 0 for background)
    """
    logger.info(f"Generating {n_events} synthetic collider events...")
    
    events_data = []
    true_labels = []
    n_anomalous = int(n_events * anomaly_fraction)
    
    for event_id in range(n_events):
        is_anomalous = event_id < n_anomalous
        
        if is_anomalous:
            # Anomalous event: unusual signatures that could indicate new physics
            n_particles = np.random.poisson(25)  # Higher multiplicity
            particles = []
            
            for _ in range(n_particles):
                # More energetic particles, different momentum distributions
                energy = np.random.exponential(150)  # Higher energy scale
                px = np.random.normal(0, 80)  # Broader momentum distribution
                py = np.random.normal(0, 80)
                pz = np.random.normal(0, 120)
                particle_type = np.random.choice([11, 13, 22, 211, 321, 2212], 
                                               p=[0.2, 0.2, 0.2, 0.3, 0.05, 0.05])  # Different composition
                
                particles.append({
                    'px': px, 'py': py, 'pz': pz, 'E': energy,
                    'type': particle_type
                })
            
            # Anomalous event-level features
            missing_et = np.random.exponential(80)  # Higher missing energy
            total_energy = sum([p['E'] for p in particles])
            jet_multiplicity = np.random.poisson(8)  # More jets
            lepton_multiplicity = np.random.poisson(3)  # More leptons
            
        else:
            # Standard Model background event
            n_particles = np.random.poisson(15)  # Typical multiplicity
            particles = []
            
            for _ in range(n_particles):
                # Typical SM particle characteristics
                energy = np.random.exponential(50)
                px = np.random.normal(0, 30)
                py = np.random.normal(0, 30)  
                pz = np.random.normal(0, 50)
                particle_type = np.random.choice([11, 13, 22, 211, 321, 2212],
                                               p=[0.1, 0.1, 0.3, 0.4, 0.05, 0.05])  # SM composition
                
                particles.append({
                    'px': px, 'py': py, 'pz': pz, 'E': energy,
                    'type': particle_type
                })
            
            # Standard event-level features
            missing_et = np.random.exponential(20)
            total_energy = sum([p['E'] for p in particles])
            jet_multiplicity = np.random.poisson(4)
            lepton_multiplicity = np.random.poisson(1)
        
        event_data = {
            'event_id': event_id,
            'particles': particles,
            'missing_et': missing_et,
            'total_energy': total_energy,
            'jet_multiplicity': jet_multiplicity,
            'lepton_multiplicity': lepton_multiplicity,
            'n_particles': len(particles)
        }
        
        events_data.append(event_data)
        true_labels.append(1 if is_anomalous else 0)
    
    true_labels = np.array(true_labels)
    logger.info(f"Generated {n_events} events: {np.sum(true_labels)} anomalous, "
               f"{n_events - np.sum(true_labels)} background")
    
    return events_data, true_labels


if __name__ == "__main__":
    # Example usage of the particle anomaly detector
    
    # Generate synthetic data
    events_data, true_labels = generate_sample_collider_data(n_events=5000, anomaly_fraction=0.1)
    
    # Initialize detector
    detector = ParticleAnomalyDetector(
        max_particles_per_event=50,
        latent_dim=16,
        hidden_dims=[256, 128, 64, 32]
    )
    
    # Prepare data for training (using only background events)
    background_events = [event for event, label in zip(events_data, true_labels) if label == 0]
    train_data, val_data = detector.prepare_data(background_events, validation_split=0.2)
    
    # Train the autoencoder
    detector.train(train_data, val_data, epochs=50, batch_size=128)
    
    # Plot training history
    detector.plot_training_history()
    
    # Detect anomalies on full dataset
    anomaly_labels, anomaly_info = detector.detect_anomalies(events_data, contamination=0.1)
    
    # Evaluate performance
    if true_labels is not None:
        auc_score = roc_auc_score(true_labels, anomaly_info['combined_scores'])
        print(f"\nPerformance Evaluation:")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, anomaly_labels))
    
    # Plot analysis
    detector.plot_anomaly_analysis(anomaly_info, true_labels)
    
    # Save the trained model
    detector.save_model("particle_anomaly_detector.pth")
    
    logger.info("Particle collision anomaly detection pipeline completed successfully!")