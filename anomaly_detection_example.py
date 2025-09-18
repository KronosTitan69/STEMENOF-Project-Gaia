"""
Comprehensive Example: Particle Collision Anomaly Detection Pipeline

This script demonstrates the complete workflow for detecting anomalous particle 
collision events using deep autoencoders, from data loading to evaluation.

Author: STEMENOF Project Gaia
Date: 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
import logging

# Import our modules
from particle_anomaly_detector import ParticleAnomalyDetector, generate_sample_collider_data
from data_loader import DarkMachinesDataLoader, DatasetValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_synthetic_data_pipeline():
    """
    Demonstrate the complete pipeline using synthetic particle collision data.
    This mimics the structure of real LHC data but with controlled anomalies.
    """
    print("="*80)
    print("SYNTHETIC DATA PIPELINE DEMONSTRATION")
    print("="*80)
    
    # Step 1: Generate synthetic particle collision data
    logger.info("Step 1: Generating synthetic collider data...")
    events_data, true_labels = generate_sample_collider_data(
        n_events=5000, 
        anomaly_fraction=0.1  # 10% anomalous events
    )
    
    print(f"Generated {len(events_data)} events:")
    print(f"  - Background events: {np.sum(true_labels == 0)}")
    print(f"  - Anomalous events: {np.sum(true_labels == 1)}")
    
    # Step 2: Initialize and configure the anomaly detector
    logger.info("Step 2: Initializing anomaly detector...")
    detector = ParticleAnomalyDetector(
        max_particles_per_event=50,  # Handle up to 50 particles per event
        latent_dim=16,               # Compressed representation dimension
        hidden_dims=[256, 128, 64, 32]  # Encoder/decoder architecture
    )
    
    # Step 3: Prepare training data (background events only)
    logger.info("Step 3: Preparing training data...")
    background_events = [event for event, label in zip(events_data, true_labels) if label == 0]
    print(f"Using {len(background_events)} background events for training")
    
    train_data, val_data = detector.prepare_data(background_events, validation_split=0.2)
    
    # Step 4: Train the autoencoder
    logger.info("Step 4: Training autoencoder...")
    detector.train(
        train_data=train_data,
        val_data=val_data,
        epochs=50,
        batch_size=128,
        learning_rate=1e-3,
        early_stopping_patience=10
    )
    
    # Step 5: Visualize training progress
    logger.info("Step 5: Visualizing training progress...")
    detector.plot_training_history()
    
    # Step 6: Detect anomalies on the full dataset
    logger.info("Step 6: Detecting anomalies...")
    anomaly_labels, anomaly_info = detector.detect_anomalies(
        events_data, 
        contamination=0.1  # Expected 10% anomalies
    )
    
    # Step 7: Evaluate performance
    logger.info("Step 7: Evaluating performance...")
    auc_score = roc_auc_score(true_labels, anomaly_info['combined_scores'])
    
    print(f"\nPERFORMANCE RESULTS:")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Detected anomalies: {anomaly_info['n_anomalies']}/{len(events_data)}")
    print(f"Detection threshold: {anomaly_info['threshold']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, anomaly_labels, 
                               target_names=['Background', 'Anomaly']))
    
    # Step 8: Visualize results
    logger.info("Step 8: Creating analysis plots...")
    detector.plot_anomaly_analysis(anomaly_info, true_labels)
    
    # Step 9: Save the trained model
    logger.info("Step 9: Saving trained model...")
    detector.save_model("trained_particle_detector.pth")
    
    return detector, anomaly_info, true_labels

def demonstrate_dark_machines_pipeline():
    """
    Demonstrate the pipeline using Dark Machines-style data format.
    """
    print("\n" + "="*80)
    print("DARK MACHINES DATA PIPELINE DEMONSTRATION") 
    print("="*80)
    
    # Step 1: Load Dark Machines style data
    logger.info("Step 1: Loading Dark Machines data...")
    dm_loader = DarkMachinesDataLoader('./data')
    events_data, labels = dm_loader.load_dark_machines_data('sample', max_events=2000)
    
    # Step 2: Validate the loaded data
    logger.info("Step 2: Validating data structure...")
    validator = DatasetValidator()
    validation_report = validator.validate_events(events_data)
    
    print(f"Dataset Validation:")
    print(f"  - Total events: {validation_report['n_events']}")
    print(f"  - Valid events: {validation_report['valid_events']}")
    print(f"  - Invalid events: {validation_report['invalid_events']}")
    
    if validation_report['statistics']:
        stats = validation_report['statistics']
        print(f"  - Avg particles per event: {stats['avg_particles_per_event']:.1f} ± {stats['std_particles_per_event']:.1f}")
        print(f"  - Particle count range: {stats['min_particles_per_event']} - {stats['max_particles_per_event']}")
    
    # Step 3: Visualize dataset overview
    logger.info("Step 3: Plotting dataset overview...")
    validator.plot_dataset_overview(events_data, labels)
    
    # Step 4: Train anomaly detector
    logger.info("Step 4: Training anomaly detector on Dark Machines data...")
    detector = ParticleAnomalyDetector(
        max_particles_per_event=30,
        latent_dim=12,
        hidden_dims=[128, 64, 32]
    )
    
    # Use background events for training (label == 0)
    if labels is not None:
        background_events = [event for event, label in zip(events_data, labels) if label == 0]
        train_data, val_data = detector.prepare_data(background_events, validation_split=0.2)
    else:
        # If no labels, use all data (unsupervised approach)
        train_data, val_data = detector.prepare_data(events_data, validation_split=0.2)
    
    detector.train(train_data, val_data, epochs=30, batch_size=64)
    
    # Step 5: Detect anomalies
    logger.info("Step 5: Detecting anomalies in Dark Machines data...")
    anomaly_labels, anomaly_info = detector.detect_anomalies(events_data, contamination=0.1)
    
    # Step 6: Evaluate if labels are available
    if labels is not None:
        auc_score = roc_auc_score(labels, anomaly_info['combined_scores'])
        print(f"\nDark Machines Performance:")
        print(f"AUC Score: {auc_score:.4f}")
        print(classification_report(labels, anomaly_labels, target_names=['Background', 'Anomaly']))
        
        detector.plot_anomaly_analysis(anomaly_info, labels)
    else:
        print(f"\nUnsupervised Results:")
        print(f"Detected {anomaly_info['n_anomalies']} anomalous events")
        detector.plot_anomaly_analysis(anomaly_info)
    
    return detector, anomaly_info

def compare_anomaly_detection_methods(events_data, true_labels):
    """
    Compare different anomaly detection approaches on the same dataset.
    """
    print("\n" + "="*80)
    print("ANOMALY DETECTION METHOD COMPARISON")
    print("="*80)
    
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    
    # Prepare data for sklearn methods
    detector = ParticleAnomalyDetector(max_particles_per_event=50)
    features = detector.preprocessor.fit_transform(events_data)
    
    methods = {}
    
    # 1. Deep Autoencoder (our method)
    logger.info("Training deep autoencoder...")
    background_events = [event for event, label in zip(events_data, true_labels) if label == 0]
    train_data, val_data = detector.prepare_data(background_events, validation_split=0.2)
    detector.train(train_data, val_data, epochs=30, batch_size=128)
    
    _, anomaly_info = detector.detect_anomalies(events_data, contamination=0.1)
    autoencoder_scores = anomaly_info['combined_scores']
    methods['Deep Autoencoder'] = autoencoder_scores
    
    # 2. Isolation Forest
    logger.info("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(features[true_labels == 0])  # Train on background only
    iso_scores = -iso_forest.decision_function(features)  # Negative for anomaly scores
    methods['Isolation Forest'] = iso_scores
    
    # 3. One-Class SVM
    logger.info("Training One-Class SVM...")
    oc_svm = OneClassSVM(gamma='scale', nu=0.1)
    oc_svm.fit(features[true_labels == 0])  # Train on background only
    svm_scores = -oc_svm.decision_function(features)  # Negative for anomaly scores
    methods['One-Class SVM'] = svm_scores
    
    # Compare performance
    print(f"\nMETHOD COMPARISON RESULTS:")
    print("-" * 60)
    
    for method_name, scores in methods.items():
        # Normalize scores for fair comparison
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        auc = roc_auc_score(true_labels, scores_norm)
        print(f"{method_name:20s}: AUC = {auc:.4f}")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for method_name, scores in methods.items():
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(true_labels, scores_norm)
        auc = roc_auc_score(true_labels, scores_norm)
        plt.plot(fpr, tpr, label=f'{method_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Anomaly Detection Method Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_anomaly_characteristics(detector, events_data, anomaly_info, true_labels):
    """
    Analyze the characteristics of detected anomalies to understand what makes them anomalous.
    """
    print("\n" + "="*80)
    print("ANOMALY CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # Get predictions
    predicted_anomalies = anomaly_info['combined_scores'] > anomaly_info['threshold']
    
    # Separate true positives, false positives, etc.
    true_positives = (true_labels == 1) & (predicted_anomalies == 1)
    false_positives = (true_labels == 0) & (predicted_anomalies == 1)
    true_negatives = (true_labels == 0) & (predicted_anomalies == 0)
    false_negatives = (true_labels == 1) & (predicted_anomalies == 0)
    
    print(f"Classification Breakdown:")
    print(f"  True Positives (correctly detected anomalies): {np.sum(true_positives)}")
    print(f"  False Positives (background labeled as anomaly): {np.sum(false_positives)}")
    print(f"  True Negatives (correctly identified background): {np.sum(true_negatives)}")
    print(f"  False Negatives (missed anomalies): {np.sum(false_negatives)}")
    
    # Analyze event characteristics
    def get_event_stats(event_indices):
        if np.sum(event_indices) == 0:
            return {}
        
        selected_events = [events_data[i] for i in range(len(events_data)) if event_indices[i]]
        
        n_particles = [len(event['particles']) for event in selected_events]
        total_energies = [event.get('total_energy', 0) for event in selected_events]
        missing_ets = [event.get('missing_et', 0) for event in selected_events]
        jet_multiplicities = [event.get('jet_multiplicity', 0) for event in selected_events]
        
        return {
            'n_particles': {'mean': np.mean(n_particles), 'std': np.std(n_particles)},
            'total_energy': {'mean': np.mean(total_energies), 'std': np.std(total_energies)},
            'missing_et': {'mean': np.mean(missing_ets), 'std': np.std(missing_ets)},
            'jet_multiplicity': {'mean': np.mean(jet_multiplicities), 'std': np.std(jet_multiplicities)}
        }
    
    # Get statistics for different categories
    tp_stats = get_event_stats(true_positives)
    fp_stats = get_event_stats(false_positives)
    tn_stats = get_event_stats(true_negatives)
    fn_stats = get_event_stats(false_negatives)
    
    # Print analysis
    categories = [
        ("True Positives", tp_stats),
        ("False Positives", fp_stats), 
        ("True Negatives", tn_stats),
        ("False Negatives", fn_stats)
    ]
    
    print(f"\nEvent Characteristics by Category:")
    print("-" * 80)
    
    for cat_name, stats in categories:
        if stats:
            print(f"\n{cat_name}:")
            for feature, values in stats.items():
                print(f"  {feature:15s}: {values['mean']:8.2f} ± {values['std']:6.2f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    features_to_plot = ['n_particles', 'total_energy', 'missing_et', 'jet_multiplicity']
    feature_labels = ['Number of Particles', 'Total Energy', 'Missing ET', 'Jet Multiplicity']
    
    for i, (feature, label) in enumerate(zip(features_to_plot, feature_labels)):
        row, col = i // 2, i % 2
        
        # Get data for true anomalies and background
        true_anomaly_data = []
        background_data = []
        
        for j, event in enumerate(events_data):
            if feature == 'n_particles':
                value = len(event['particles'])
            else:
                value = event.get(feature, 0)
                
            if true_labels[j] == 1:
                true_anomaly_data.append(value)
            else:
                background_data.append(value)
        
        # Plot histograms
        axes[row, col].hist(background_data, bins=30, alpha=0.7, label='Background', density=True)
        axes[row, col].hist(true_anomaly_data, bins=30, alpha=0.7, label='True Anomalies', density=True)
        axes[row, col].set_xlabel(label)
        axes[row, col].set_ylabel('Density')
        axes[row, col].set_title(f'{label} Distribution')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the complete demonstration pipeline.
    """
    print("PARTICLE COLLISION ANOMALY DETECTION PIPELINE")
    print("Advanced Deep Learning for High Energy Physics")
    print("=" * 80)
    
    # Run synthetic data demonstration
    detector, anomaly_info, true_labels = demonstrate_synthetic_data_pipeline()
    
    # Run Dark Machines demonstration
    dm_detector, dm_anomaly_info = demonstrate_dark_machines_pipeline()
    
    # Compare different methods
    events_data, _ = generate_sample_collider_data(n_events=2000, anomaly_fraction=0.1)
    compare_anomaly_detection_methods(events_data, _)
    
    # Analyze anomaly characteristics
    analyze_anomaly_characteristics(detector, events_data, anomaly_info, _)
    
    print("\n" + "="*80)
    print("PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey Files Generated:")
    print("  - trained_particle_detector.pth: Trained autoencoder model")
    print("  - best_autoencoder.pth: Best model checkpoint")
    print("  - ./data/sample_data.h5: Sample Dark Machines format data")
    print("\nThe pipeline demonstrated:")
    print("  ✓ Synthetic particle collision data generation")
    print("  ✓ Real-world data format compatibility (Dark Machines)")
    print("  ✓ Deep autoencoder training for anomaly detection")
    print("  ✓ Model evaluation and performance metrics")
    print("  ✓ Comparison with traditional ML methods")
    print("  ✓ Anomaly characteristic analysis")

if __name__ == "__main__":
    main()