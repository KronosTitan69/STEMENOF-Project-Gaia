#!/usr/bin/env python3
"""
Command Line Interface for Particle Collision Anomaly Detection

This script provides a simple command-line interface for training and using
the particle collision anomaly detection system.

Usage:
    python cli.py --help
    python cli.py train --data synthetic --events 5000
    python cli.py detect --model trained_model.pth --data synthetic --events 1000

Author: STEMENOF Project Gaia
Date: 2025
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(args):
    """Train a new anomaly detection model"""
    from particle_anomaly_detector import ParticleAnomalyDetector, generate_sample_collider_data
    from data_loader import DarkMachinesDataLoader
    
    print("🚀 Training Particle Collision Anomaly Detection Model")
    print("=" * 60)
    
    # Load or generate data
    if args.data == 'synthetic':
        print(f"📊 Generating {args.events} synthetic events...")
        events_data, true_labels = generate_sample_collider_data(
            n_events=args.events, 
            anomaly_fraction=args.anomaly_fraction
        )
        # Use only background events for training
        background_events = [event for event, label in zip(events_data, true_labels) if label == 0]
        
    elif args.data == 'dark_machines':
        print(f"📊 Loading Dark Machines data...")
        dm_loader = DarkMachinesDataLoader(args.data_dir)
        events_data, labels = dm_loader.load_dark_machines_data('sample', max_events=args.events)
        if labels is not None:
            background_events = [event for event, label in zip(events_data, labels) if label == 0]
        else:
            background_events = events_data  # Assume all are background for unsupervised training
    
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    print(f"   Using {len(background_events)} background events for training")
    
    # Initialize detector
    detector = ParticleAnomalyDetector(
        max_particles_per_event=args.max_particles,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    )
    
    # Prepare and train
    train_data, val_data = detector.prepare_data(background_events, validation_split=0.2)
    
    print(f"🧠 Training autoencoder for {args.epochs} epochs...")
    detector.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save model
    output_path = args.output or "trained_particle_detector.pth"
    detector.save_model(output_path)
    
    print(f"✅ Model training completed!")
    print(f"💾 Model saved to: {output_path}")
    
    return detector

def detect_anomalies(args):
    """Detect anomalies using a trained model"""
    from particle_anomaly_detector import ParticleAnomalyDetector, generate_sample_collider_data
    from data_loader import DarkMachinesDataLoader
    from sklearn.metrics import roc_auc_score, classification_report
    
    print("🔍 Detecting Particle Collision Anomalies")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    detector = ParticleAnomalyDetector()
    detector.load_model(args.model)
    print(f"📥 Loaded model from: {args.model}")
    
    # Load data
    if args.data == 'synthetic':
        print(f"📊 Generating {args.events} synthetic events for testing...")
        events_data, true_labels = generate_sample_collider_data(
            n_events=args.events,
            anomaly_fraction=args.anomaly_fraction
        )
        
    elif args.data == 'dark_machines':
        print(f"📊 Loading Dark Machines data for testing...")
        dm_loader = DarkMachinesDataLoader(args.data_dir)
        events_data, true_labels = dm_loader.load_dark_machines_data('sample', max_events=args.events)
        
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    print(f"   Loaded {len(events_data)} events")
    
    # Detect anomalies
    print("🔍 Running anomaly detection...")
    anomaly_labels, anomaly_info = detector.detect_anomalies(
        events_data, 
        contamination=args.contamination
    )
    
    # Print results
    print(f"\n📈 DETECTION RESULTS:")
    print(f"   Total events analyzed: {len(events_data)}")
    print(f"   Anomalies detected: {anomaly_info['n_anomalies']}")
    print(f"   Detection rate: {anomaly_info['n_anomalies']/len(events_data)*100:.2f}%")
    print(f"   Anomaly threshold: {anomaly_info['threshold']:.6f}")
    
    # Evaluate if true labels are available
    if true_labels is not None:
        auc_score = roc_auc_score(true_labels, anomaly_info['combined_scores'])
        print(f"\n📊 PERFORMANCE EVALUATION:")
        print(f"   AUC Score: {auc_score:.4f}")
        print(f"   True anomalies in data: {sum(true_labels)}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(true_labels, anomaly_labels, target_names=['Background', 'Anomaly']))
    
    # Save results if requested
    if args.output:
        import numpy as np
        import pandas as pd
        
        results_df = pd.DataFrame({
            'event_id': range(len(events_data)),
            'anomaly_score': anomaly_info['combined_scores'],
            'reconstruction_error': anomaly_info['reconstruction_errors'],
            'latent_distance': anomaly_info['latent_distances'],
            'predicted_anomaly': anomaly_labels,
            'true_anomaly': true_labels if true_labels is not None else [-1] * len(events_data)
        })
        
        results_df.to_csv(args.output, index=False)
        print(f"💾 Results saved to: {args.output}")
    
    return anomaly_labels, anomaly_info

def visualize_results(args):
    """Create visualization of anomaly detection results"""
    print("📊 Creating Anomaly Detection Visualizations")
    print("=" * 50)
    
    # Import the visualization demo
    from demo_visualization import create_comprehensive_demo
    
    # Run the comprehensive demo
    detector, anomaly_info, true_labels = create_comprehensive_demo()
    
    print("✅ Visualization completed!")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Particle Collision Anomaly Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model on synthetic data
  python cli.py train --data synthetic --events 5000 --epochs 50
  
  # Detect anomalies using trained model
  python cli.py detect --model trained_particle_detector.pth --data synthetic --events 1000
  
  # Create comprehensive visualization
  python cli.py visualize
  
  # Train with custom architecture
  python cli.py train --data synthetic --latent-dim 32 --hidden-dims 512 256 128 64
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new anomaly detection model')
    train_parser.add_argument('--data', choices=['synthetic', 'dark_machines'], default='synthetic',
                             help='Data source for training')
    train_parser.add_argument('--events', type=int, default=5000,
                             help='Number of events to use for training')
    train_parser.add_argument('--anomaly-fraction', type=float, default=0.1,
                             help='Fraction of anomalous events in synthetic data')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=128,
                             help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-3,
                             help='Learning rate for optimization')
    train_parser.add_argument('--max-particles', type=int, default=50,
                             help='Maximum particles per event to consider')
    train_parser.add_argument('--latent-dim', type=int, default=16,
                             help='Latent space dimensionality')
    train_parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64, 32],
                             help='Hidden layer dimensions')
    train_parser.add_argument('--data-dir', default='./data',
                             help='Directory for data files')
    train_parser.add_argument('--output', default='trained_particle_detector.pth',
                             help='Output path for trained model')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect anomalies using trained model')
    detect_parser.add_argument('--model', required=True,
                              help='Path to trained model file')
    detect_parser.add_argument('--data', choices=['synthetic', 'dark_machines'], default='synthetic',
                              help='Data source for detection')
    detect_parser.add_argument('--events', type=int, default=1000,
                              help='Number of events to analyze')
    detect_parser.add_argument('--contamination', type=float, default=0.1,
                              help='Expected fraction of anomalous events')
    detect_parser.add_argument('--anomaly-fraction', type=float, default=0.1,
                              help='Fraction of anomalous events in synthetic data')
    detect_parser.add_argument('--data-dir', default='./data',
                              help='Directory for data files')
    detect_parser.add_argument('--output', help='Output CSV file for results')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create comprehensive visualizations')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'detect':
        detect_anomalies(args)
    elif args.command == 'visualize':
        visualize_results(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()