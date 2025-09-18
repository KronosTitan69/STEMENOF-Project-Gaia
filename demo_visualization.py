"""
Visual Demonstration of Particle Collision Anomaly Detection

This script creates visual demonstrations of the deep autoencoder system
for particle collision anomaly detection with comprehensive plots and analysis.

Author: STEMENOF Project Gaia
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import logging

from particle_anomaly_detector import ParticleAnomalyDetector, generate_sample_collider_data

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_comprehensive_demo():
    """Create a comprehensive visual demonstration of the anomaly detection system"""
    
    print("🚀 PARTICLE COLLISION ANOMALY DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # 1. Generate synthetic particle collision data
    print("📊 Generating synthetic particle collision data...")
    events_data, true_labels = generate_sample_collider_data(n_events=2000, anomaly_fraction=0.1)
    
    print(f"   Generated {len(events_data)} events:")
    print(f"   • Background events: {np.sum(true_labels == 0)}")
    print(f"   • Anomalous events: {np.sum(true_labels == 1)}")
    
    # 2. Initialize and train the detector
    print("\n🧠 Training deep autoencoder...")
    detector = ParticleAnomalyDetector(
        max_particles_per_event=40,
        latent_dim=16,
        hidden_dims=[256, 128, 64, 32]
    )
    
    # Prepare training data (background only)
    background_events = [event for event, label in zip(events_data, true_labels) if label == 0]
    train_data, val_data = detector.prepare_data(background_events, validation_split=0.2)
    
    # Train the model
    detector.train(train_data, val_data, epochs=30, batch_size=128)
    
    # 3. Detect anomalies
    print("\n🔍 Detecting anomalies...")
    anomaly_labels, anomaly_info = detector.detect_anomalies(events_data, contamination=0.1)
    
    # 4. Create comprehensive visualizations
    print("\n📈 Creating visualizations...")
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Training Loss Curves
    ax1 = plt.subplot(3, 4, 1)
    plt.plot(detector.train_losses, label='Training Loss', linewidth=2, alpha=0.8)
    plt.plot(detector.val_losses, label='Validation Loss', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Reconstruction Error Distribution
    ax2 = plt.subplot(3, 4, 2)
    background_errors = anomaly_info['reconstruction_errors'][true_labels == 0]
    anomaly_errors = anomaly_info['reconstruction_errors'][true_labels == 1]
    
    plt.hist(background_errors, bins=50, alpha=0.7, label='Background', density=True, color='skyblue')
    plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomalies', density=True, color='salmon')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 3. ROC Curve
    ax3 = plt.subplot(3, 4, 3)
    fpr, tpr, thresholds = roc_curve(true_labels, anomaly_info['combined_scores'])
    auc_score = roc_auc_score(true_labels, anomaly_info['combined_scores'])
    
    plt.plot(fpr, tpr, linewidth=3, label=f'Deep Autoencoder (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(3, 4, 4)
    cm = confusion_matrix(true_labels, anomaly_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Anomaly'],
                yticklabels=['Background', 'Anomaly'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 5. Combined Anomaly Score Distribution
    ax5 = plt.subplot(3, 4, 5)
    background_scores = anomaly_info['combined_scores'][true_labels == 0]
    anomaly_scores = anomaly_info['combined_scores'][true_labels == 1]
    
    plt.hist(background_scores, bins=50, alpha=0.7, label='Background', density=True, color='skyblue')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomalies', density=True, color='salmon')
    plt.axvline(anomaly_info['threshold'], color='red', linestyle='--', linewidth=2, 
                label=f'Threshold: {anomaly_info["threshold"]:.3f}')
    plt.xlabel('Combined Anomaly Score')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Event Characteristics Comparison - Number of Particles
    ax6 = plt.subplot(3, 4, 6)
    bg_n_particles = [len(events_data[i]['particles']) for i in range(len(events_data)) if true_labels[i] == 0]
    an_n_particles = [len(events_data[i]['particles']) for i in range(len(events_data)) if true_labels[i] == 1]
    
    plt.hist(bg_n_particles, bins=30, alpha=0.7, label='Background', density=True, color='skyblue')
    plt.hist(an_n_particles, bins=30, alpha=0.7, label='Anomalies', density=True, color='salmon')
    plt.xlabel('Number of Particles')
    plt.ylabel('Density')
    plt.title('Particle Multiplicity', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Event Characteristics Comparison - Total Energy
    ax7 = plt.subplot(3, 4, 7)
    bg_energy = [events_data[i]['total_energy'] for i in range(len(events_data)) if true_labels[i] == 0]
    an_energy = [events_data[i]['total_energy'] for i in range(len(events_data)) if true_labels[i] == 1]
    
    plt.hist(bg_energy, bins=30, alpha=0.7, label='Background', density=True, color='skyblue')
    plt.hist(an_energy, bins=30, alpha=0.7, label='Anomalies', density=True, color='salmon')
    plt.xlabel('Total Energy')
    plt.ylabel('Density')
    plt.title('Total Event Energy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Event Characteristics Comparison - Missing ET
    ax8 = plt.subplot(3, 4, 8)
    bg_met = [events_data[i]['missing_et'] for i in range(len(events_data)) if true_labels[i] == 0]
    an_met = [events_data[i]['missing_et'] for i in range(len(events_data)) if true_labels[i] == 1]
    
    plt.hist(bg_met, bins=30, alpha=0.7, label='Background', density=True, color='skyblue')
    plt.hist(an_met, bins=30, alpha=0.7, label='Anomalies', density=True, color='salmon')
    plt.xlabel('Missing Transverse Energy')
    plt.ylabel('Density')
    plt.title('Missing ET Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Latent Space Visualization (2D projection)
    ax9 = plt.subplot(3, 4, 9)
    # Get latent representations
    features = detector.preprocessor.transform(events_data)
    import torch
    features_tensor = torch.FloatTensor(features).to(detector.device)
    
    with torch.no_grad():
        _, latent = detector.model(features_tensor)
        latent_np = latent.cpu().numpy()
    
    # Project to 2D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_np)
    
    # Plot latent space
    bg_indices = true_labels == 0
    an_indices = true_labels == 1
    
    plt.scatter(latent_2d[bg_indices, 0], latent_2d[bg_indices, 1], 
                alpha=0.6, s=30, label='Background', color='skyblue')
    plt.scatter(latent_2d[an_indices, 0], latent_2d[an_indices, 1], 
                alpha=0.8, s=50, label='Anomalies', color='salmon', marker='^')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Latent Space Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Performance Metrics Summary
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(true_labels, anomaly_labels)
    recall = recall_score(true_labels, anomaly_labels)
    f1 = f1_score(true_labels, anomaly_labels)
    
    metrics_text = f"""
    PERFORMANCE METRICS
    
    AUC Score: {auc_score:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1 Score: {f1:.4f}
    
    Threshold: {anomaly_info['threshold']:.4f}
    
    DETECTION SUMMARY
    Total Events: {len(events_data)}
    True Anomalies: {np.sum(true_labels)}
    Detected Anomalies: {anomaly_info['n_anomalies']}
    
    True Positives: {np.sum((true_labels == 1) & (anomaly_labels == 1))}
    False Positives: {np.sum((true_labels == 0) & (anomaly_labels == 1))}
    False Negatives: {np.sum((true_labels == 1) & (anomaly_labels == 0))}
    """
    
    plt.text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 11. Architecture Diagram (text representation)
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    arch_text = f"""
    AUTOENCODER ARCHITECTURE
    
    Input Layer: {detector.model.input_dim} features
    ↓
    Encoder:
    {' → '.join(map(str, detector.hidden_dims))}
    ↓
    Latent Space: {detector.latent_dim} dimensions
    ↓
    Decoder:
    {' → '.join(map(str, reversed(detector.hidden_dims)))}
    ↓
    Output Layer: {detector.model.input_dim} features
    
    ANOMALY DETECTION:
    • Reconstruction Error (70%)
    • Latent Distance (30%)
    """
    
    plt.text(0.1, 0.9, arch_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 12. Feature Importance (simplified)
    ax12 = plt.subplot(3, 4, 12)
    
    # Calculate feature importance based on reconstruction error contribution
    feature_importance = np.mean(np.abs(features - detector.model(features_tensor)[0].detach().cpu().numpy()), axis=0)
    top_features = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    plt.barh(range(len(top_features)), feature_importance[top_features], color='steelblue', alpha=0.7)
    plt.ylabel('Feature Index')
    plt.xlabel('Average Reconstruction Error')
    plt.title('Top Contributing Features', fontsize=14, fontweight='bold')
    plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_features])
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout and add main title
    plt.tight_layout()
    plt.suptitle('Particle Collision Anomaly Detection: Deep Autoencoder Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.94)
    plt.show()
    
    # Print summary
    print(f"\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print(f"🎯 AUC Score: {auc_score:.4f}")
    print(f"🔍 Detected {anomaly_info['n_anomalies']} anomalous events out of {len(events_data)}")
    print(f"📊 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return detector, anomaly_info, true_labels

if __name__ == "__main__":
    # Run the comprehensive demonstration
    detector, anomaly_info, true_labels = create_comprehensive_demo()