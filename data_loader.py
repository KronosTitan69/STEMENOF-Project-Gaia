"""
Data Loader for Particle Physics Datasets

This module provides utilities to load and process real particle physics datasets
including LHC data and Dark Machines Anomaly Score Challenge datasets.

Author: STEMENOF Project Gaia
Date: 2025
"""

import numpy as np
import pandas as pd
import h5py
import os
from typing import List, Dict, Tuple, Optional
import logging
import requests
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

class LHCDataLoader:
    """
    Data loader for LHC-style particle physics datasets.
    Supports various formats including HDF5, CSV, and ROOT files.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.supported_formats = ['.h5', '.hdf5', '.csv', '.json']
        
    def load_events(self, max_events: Optional[int] = None) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Load events from the dataset file.
        
        Args:
            max_events: Maximum number of events to load (None for all)
            
        Returns:
            events_data: List of event dictionaries
            labels: Labels if available (None otherwise)
        """
        file_extension = os.path.splitext(self.data_path)[1].lower()
        
        if file_extension in ['.h5', '.hdf5']:
            return self._load_hdf5(max_events)
        elif file_extension == '.csv':
            return self._load_csv(max_events)
        elif file_extension == '.json':
            return self._load_json(max_events)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_hdf5(self, max_events: Optional[int] = None) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Load events from HDF5 format"""
        logger.info(f"Loading HDF5 data from {self.data_path}")
        
        events_data = []
        labels = None
        
        with h5py.File(self.data_path, 'r') as f:
            # Explore HDF5 structure
            logger.info(f"HDF5 keys: {list(f.keys())}")
            
            # Common HDF5 structures for particle physics
            if 'events' in f:
                events_group = f['events']
                n_events = len(events_group) if max_events is None else min(max_events, len(events_group))
                
                for i in tqdm(range(n_events), desc="Loading events"):
                    event_data = self._parse_hdf5_event(events_group[str(i)])
                    events_data.append(event_data)
                    
            elif 'particles' in f and 'event_info' in f:
                # Structure with separate particle and event info arrays
                particles_data = f['particles'][:]
                event_info = f['event_info'][:]
                
                n_events = len(event_info) if max_events is None else min(max_events, len(event_info))
                
                for i in tqdm(range(n_events), desc="Loading events"):
                    event_data = self._parse_structured_event(particles_data, event_info, i)
                    events_data.append(event_data)
                    
            # Load labels if available
            if 'labels' in f:
                labels = f['labels'][:n_events] if max_events else f['labels'][:]
                
        logger.info(f"Loaded {len(events_data)} events")
        return events_data, labels
    
    def _parse_hdf5_event(self, event_group) -> Dict:
        """Parse a single event from HDF5 group structure"""
        event_data = {'particles': []}
        
        # Extract particle information
        if 'particles' in event_group:
            particles_data = event_group['particles']
            
            for particle_data in particles_data:
                particle = {
                    'px': float(particle_data.get('px', 0)),
                    'py': float(particle_data.get('py', 0)),
                    'pz': float(particle_data.get('pz', 0)),
                    'E': float(particle_data.get('E', 0)),
                    'type': int(particle_data.get('pdg_id', 0))
                }
                event_data['particles'].append(particle)
        
        # Extract event-level information
        event_data.update({
            'missing_et': float(event_group.get('missing_et', 0)),
            'total_energy': float(event_group.get('total_energy', 0)),
            'jet_multiplicity': int(event_group.get('n_jets', 0)),
            'lepton_multiplicity': int(event_group.get('n_leptons', 0)),
            'event_id': int(event_group.get('event_id', 0))
        })
        
        return event_data
    
    def _parse_structured_event(self, particles_data: np.ndarray, 
                               event_info: np.ndarray, event_idx: int) -> Dict:
        """Parse event from structured array format"""
        # Find particles belonging to this event
        event_particles = particles_data[particles_data['event_id'] == event_idx]
        
        particles = []
        for particle in event_particles:
            particles.append({
                'px': float(particle['px']),
                'py': float(particle['py']),
                'pz': float(particle['pz']),
                'E': float(particle['E']),
                'type': int(particle.get('pdg_id', 0))
            })
        
        # Event-level information
        event_info_row = event_info[event_idx]
        event_data = {
            'particles': particles,
            'missing_et': float(event_info_row.get('missing_et', 0)),
            'total_energy': float(event_info_row.get('total_energy', 0)),
            'jet_multiplicity': int(event_info_row.get('n_jets', 0)),
            'lepton_multiplicity': int(event_info_row.get('n_leptons', 0)),
            'event_id': event_idx
        }
        
        return event_data
    
    def _load_csv(self, max_events: Optional[int] = None) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Load events from CSV format"""
        logger.info(f"Loading CSV data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        if max_events:
            df = df.head(max_events)
            
        events_data = []
        
        # Assume CSV has flattened particle features
        # This is a simplified parser - real datasets may need custom parsing
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading events"):
            particles = self._parse_csv_particles(row)
            
            event_data = {
                'particles': particles,
                'missing_et': row.get('missing_et', 0),
                'total_energy': row.get('total_energy', 0),
                'jet_multiplicity': row.get('n_jets', 0),
                'lepton_multiplicity': row.get('n_leptons', 0),
                'event_id': row.get('event_id', len(events_data))
            }
            
            events_data.append(event_data)
        
        # Extract labels if present
        labels = df['label'].values if 'label' in df.columns else None
        
        return events_data, labels
    
    def _parse_csv_particles(self, row: pd.Series) -> List[Dict]:
        """Parse particle information from flattened CSV row"""
        particles = []
        
        # Look for particle columns (assuming naming convention like px_0, py_0, etc.)
        particle_idx = 0
        while f'px_{particle_idx}' in row:
            if not pd.isna(row[f'px_{particle_idx}']):
                particle = {
                    'px': float(row[f'px_{particle_idx}']),
                    'py': float(row[f'py_{particle_idx}']),
                    'pz': float(row[f'pz_{particle_idx}']),
                    'E': float(row[f'E_{particle_idx}']),
                    'type': int(row.get(f'type_{particle_idx}', 0))
                }
                particles.append(particle)
            particle_idx += 1
            
        return particles
    
    def _load_json(self, max_events: Optional[int] = None) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Load events from JSON format"""
        logger.info(f"Loading JSON data from {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            
        events_data = data.get('events', data)  # Handle different JSON structures
        if max_events:
            events_data = events_data[:max_events]
            
        labels = np.array(data.get('labels')) if 'labels' in data else None
        
        return events_data, labels


class DarkMachinesDataLoader:
    """
    Data loader specifically for Dark Machines Anomaly Score Challenge datasets.
    Handles the specific format and structure of these datasets.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.challenge_datasets = {
            'R&D': 'RnD_data.h5',
            'LHC_Olympics': 'lhco_data.h5', 
            'Background': 'background_data.h5',
            'Signal': 'signal_data.h5'
        }
    
    def download_sample_data(self, dataset_name: str = 'sample'):
        """
        Download sample Dark Machines data for demonstration.
        In a real implementation, this would fetch from the official dataset.
        """
        # This would typically download from:
        # https://www.darkmachines.org/
        # For demonstration, we'll create a sample dataset
        
        logger.info(f"Creating sample dataset for demonstration...")
        
        # Generate sample data in Dark Machines format
        sample_events = []
        sample_labels = []
        
        for i in range(1000):
            # Generate event in Dark Machines format
            n_particles = np.random.poisson(10)
            particles = []
            
            for j in range(n_particles):
                # Dark Machines typically uses (pt, eta, phi, mass, pdg_id)
                pt = np.random.exponential(50)
                eta = np.random.normal(0, 2)
                phi = np.random.uniform(-np.pi, np.pi) 
                mass = np.random.exponential(5)
                pdg_id = np.random.choice([11, 13, 22, 211, 321])
                
                particles.append([pt, eta, phi, mass, pdg_id])
            
            sample_events.append(particles)
            sample_labels.append(np.random.choice([0, 1], p=[0.9, 0.1]))  # 10% anomalies
        
        # Save as HDF5
        sample_path = os.path.join(self.data_dir, f'{dataset_name}_data.h5')
        os.makedirs(self.data_dir, exist_ok=True)
        
        with h5py.File(sample_path, 'w') as f:
            # Store events with variable length
            dt = h5py.special_dtype(vlen=np.dtype('float32'))
            events_dataset = f.create_dataset('events', (len(sample_events),), dtype=dt)
            
            for i, event in enumerate(sample_events):
                events_dataset[i] = np.array(event, dtype=np.float32).flatten()
                
            f.create_dataset('labels', data=np.array(sample_labels))
            f.create_dataset('n_particles', data=[len(event) for event in sample_events])
        
        logger.info(f"Sample dataset saved to {sample_path}")
        return sample_path
    
    def load_dark_machines_data(self, dataset_name: str = 'sample', 
                               max_events: Optional[int] = None) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Load Dark Machines challenge data.
        
        Args:
            dataset_name: Name of the dataset to load
            max_events: Maximum number of events to load
            
        Returns:
            events_data: List of event dictionaries
            labels: Labels if available
        """
        data_file = os.path.join(self.data_dir, f'{dataset_name}_data.h5')
        
        if not os.path.exists(data_file):
            logger.warning(f"Data file {data_file} not found. Creating sample data...")
            data_file = self.download_sample_data(dataset_name)
        
        events_data = []
        labels = None
        
        with h5py.File(data_file, 'r') as f:
            events_raw = f['events'][:]
            n_particles_per_event = f['n_particles'][:]
            
            if 'labels' in f:
                labels = f['labels'][:]
            
            n_events = len(events_raw) if max_events is None else min(max_events, len(events_raw))
            
            start_idx = 0
            for i in tqdm(range(n_events), desc="Loading Dark Machines events"):
                n_particles = n_particles_per_event[i]
                
                # Extract particles for this event (pt, eta, phi, mass, pdg_id format)
                event_particles_flat = events_raw[i]
                event_particles = event_particles_flat.reshape(-1, 5)[:n_particles]
                
                particles = []
                total_energy = 0
                
                for particle_data in event_particles:
                    pt, eta, phi, mass, pdg_id = particle_data
                    
                    # Convert to Cartesian coordinates (px, py, pz, E)
                    px = pt * np.cos(phi)
                    py = pt * np.sin(phi)
                    pz = pt * np.sinh(eta)
                    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
                    
                    particles.append({
                        'px': float(px),
                        'py': float(py), 
                        'pz': float(pz),
                        'E': float(E),
                        'type': int(pdg_id),
                        'pt': float(pt),
                        'eta': float(eta),
                        'phi': float(phi),
                        'mass': float(mass)
                    })
                    
                    total_energy += E
                
                # Calculate event-level variables
                missing_et = np.random.exponential(20)  # Simplified - would be calculated from actual data
                jet_multiplicity = max(1, len([p for p in particles if abs(p['type']) in [1, 2, 3, 4, 5, 21]]))
                lepton_multiplicity = len([p for p in particles if abs(p['type']) in [11, 13, 15]])
                
                event_data = {
                    'particles': particles,
                    'missing_et': missing_et,
                    'total_energy': total_energy,
                    'jet_multiplicity': jet_multiplicity,
                    'lepton_multiplicity': lepton_multiplicity,
                    'event_id': i,
                    'n_particles': n_particles
                }
                
                events_data.append(event_data)
        
        if labels is not None and max_events:
            labels = labels[:max_events]
            
        logger.info(f"Loaded {len(events_data)} Dark Machines events")
        return events_data, labels


class DatasetValidator:
    """
    Utility class to validate and analyze loaded particle physics datasets.
    """
    
    @staticmethod
    def validate_events(events_data: List[Dict]) -> Dict:
        """
        Validate the structure and content of loaded events.
        
        Returns:
            validation_report: Dictionary with validation results
        """
        report = {
            'n_events': len(events_data),
            'valid_events': 0,
            'invalid_events': 0,
            'validation_errors': [],
            'statistics': {}
        }
        
        required_fields = ['particles', 'missing_et', 'total_energy', 'jet_multiplicity']
        particle_fields = ['px', 'py', 'pz', 'E', 'type']
        
        n_particles_list = []
        energy_list = []
        
        for i, event in enumerate(events_data):
            is_valid = True
            
            # Check required event fields
            for field in required_fields:
                if field not in event:
                    report['validation_errors'].append(f"Event {i}: Missing field '{field}'")
                    is_valid = False
            
            # Check particles structure
            if 'particles' in event:
                particles = event['particles']
                n_particles_list.append(len(particles))
                
                for j, particle in enumerate(particles):
                    for pfield in particle_fields:
                        if pfield not in particle:
                            report['validation_errors'].append(
                                f"Event {i}, Particle {j}: Missing field '{pfield}'"
                            )
                            is_valid = False
                            
                    # Check for valid energy values
                    if 'E' in particle and particle['E'] < 0:
                        report['validation_errors'].append(
                            f"Event {i}, Particle {j}: Negative energy"
                        )
                        is_valid = False
                        
                energy_list.append(event.get('total_energy', 0))
            
            if is_valid:
                report['valid_events'] += 1
            else:
                report['invalid_events'] += 1
        
        # Calculate statistics
        if n_particles_list:
            report['statistics'] = {
                'avg_particles_per_event': np.mean(n_particles_list),
                'std_particles_per_event': np.std(n_particles_list),
                'min_particles_per_event': np.min(n_particles_list),
                'max_particles_per_event': np.max(n_particles_list),
                'avg_event_energy': np.mean(energy_list),
                'std_event_energy': np.std(energy_list)
            }
        
        return report
    
    @staticmethod
    def plot_dataset_overview(events_data: List[Dict], labels: Optional[np.ndarray] = None):
        """Plot overview statistics of the loaded dataset"""
        import matplotlib.pyplot as plt
        
        # Extract statistics
        n_particles = [len(event['particles']) for event in events_data]
        total_energies = [event.get('total_energy', 0) for event in events_data]
        missing_ets = [event.get('missing_et', 0) for event in events_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Particle multiplicity
        axes[0, 0].hist(n_particles, bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('Number of Particles per Event')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Particle Multiplicity Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total energy
        axes[0, 1].hist(total_energies, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Total Event Energy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Total Energy Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Missing transverse energy
        axes[1, 0].hist(missing_ets, bins=50, alpha=0.7)
        axes[1, 0].set_xlabel('Missing Transverse Energy')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Missing ET Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Label distribution (if available)
        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            axes[1, 1].bar(unique, counts, alpha=0.7)
            axes[1, 1].set_xlabel('Label')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Label Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Labels Available', 
                           ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage of data loaders
    
    # Test Dark Machines data loader
    dm_loader = DarkMachinesDataLoader('./data')
    events_data, labels = dm_loader.load_dark_machines_data('sample', max_events=1000)
    
    # Validate the loaded data
    validator = DatasetValidator()
    validation_report = validator.validate_events(events_data)
    
    print("Dataset Validation Report:")
    print(f"Total events: {validation_report['n_events']}")
    print(f"Valid events: {validation_report['valid_events']}")
    print(f"Invalid events: {validation_report['invalid_events']}")
    
    if validation_report['statistics']:
        stats = validation_report['statistics']
        print(f"\nStatistics:")
        print(f"Average particles per event: {stats['avg_particles_per_event']:.2f} ± {stats['std_particles_per_event']:.2f}")
        print(f"Particle count range: {stats['min_particles_per_event']} - {stats['max_particles_per_event']}")
        print(f"Average event energy: {stats['avg_event_energy']:.2f} ± {stats['std_event_energy']:.2f}")
    
    # Plot dataset overview
    validator.plot_dataset_overview(events_data, labels)