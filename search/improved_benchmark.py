#!/usr/bin/env python
"""
Improved benchmark runner for BitBIRCH IVF search.

Features:
- Verbose mode with progress bars
- Save intermediary files and results
- Run specific benchmark stages
- Consistent fingerprint size (2048 bits)
- Multiple search methods comparison

Usage:
    python improved_benchmark.py --dataset data/chembl_33_np.smi [options]
"""

import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("tqdm not available, falling back to standard progress reporting")
    TQDM_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.ivf_index import IVFIndex
from search.utils import load_smiles_file, generate_fingerprints
from search.similarity_engines import RDKitEngine, FPSim2Engine


class StagedBenchmark:
    """
    Benchmark runner with support for stages and intermediary file saving.
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = 'search/results',
        fp_type: str = 'morgan',
        fp_size: int = 2048,
        radius: int = 2,
        threshold: float = 0.65,
        verbose: bool = True
    ):
        """Initialize the benchmark."""
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.fp_type = fp_type
        self.fp_size = fp_size
        self.radius = radius
        self.threshold = threshold
        self.verbose = verbose
        
        # Set up fingerprint parameters
        if self.fp_type.lower() == 'morgan':
            self.fp_params = {'radius': self.radius, 'nBits': self.fp_size}
        else:
            self.fp_params = {'fpSize': self.fp_size}
            
        # Create output directories
        self.data_dir = os.path.join(self.output_dir, 'data')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize data structures
        self.smiles = []
        self.fingerprints = None  # NumPy array format
        self.fingerprints_rdkit = None  # RDKit ExplicitBitVect format
        self.query_indices = None
        self.query_fps = None
        self.query_fps_rdkit = None
        self.ground_truth = {}
        
        # Paths for saved files
        self.fps_path = os.path.join(self.data_dir, 'fingerprints.npy')
        self.fps_rdkit_path = os.path.join(self.data_dir, 'fingerprints_rdkit.pickle')
        self.smiles_path = os.path.join(self.data_dir, 'smiles.json')
        self.queries_path = os.path.join(self.data_dir, 'queries.json')
        self.ground_truth_path = os.path.join(self.data_dir, 'ground_truth.pickle')
        self.results_path = os.path.join(self.data_dir, 'benchmark_results.json')
        self.ivf_index_path = os.path.join(self.data_dir, 'ivf_index.pickle')
        
        # Load dataset name for reporting
        self.dataset_name = os.path.basename(dataset_path)
        
    def log(self, message: str) -> None:
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
            
    def load_data(self, force_reload: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Load SMILES and generate fingerprints with caching."""
        # Check if we can load from cache
        if (not force_reload and 
            os.path.exists(self.fps_path) and 
            os.path.exists(self.smiles_path)):
            
            self.log(f"Loading cached NumPy fingerprints from {self.fps_path}")
            self.fingerprints = np.load(self.fps_path)
            
            # Try to load RDKit fingerprints if available
            if os.path.exists(self.fps_rdkit_path):
                self.log(f"Loading cached RDKit fingerprints from {self.fps_rdkit_path}")
                with open(self.fps_rdkit_path, 'rb') as f:
                    self.fingerprints_rdkit = pickle.load(f)
            else:
                self.log("RDKit fingerprints cache not found, will regenerate both formats")
                # Clear the cache and regenerate
                force_reload = True
            
            if not force_reload:
                self.log(f"Loading cached SMILES from {self.smiles_path}")
                with open(self.smiles_path, 'r') as f:
                    self.smiles = json.load(f)
                    
                self.log(f"Loaded {len(self.smiles)} molecules from cache")
                return self.fingerprints, self.smiles
        
        # Load from file
        self.log(f"Loading SMILES from {self.dataset_path}")
        self.smiles = load_smiles_file(self.dataset_path)
        self.log(f"Loaded {len(self.smiles)} SMILES")
        
        # Generate fingerprints with progress reporting
        self.log(f"Generating {self.fp_type} fingerprints ({self.fp_size} bits)")
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdFingerprintGenerator
        
        # Silence RDKit warnings
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        # Convert SMILES to molecules with progress reporting
        if TQDM_AVAILABLE:
            mols = []
            for smiles in tqdm(self.smiles, desc="Converting SMILES to RDKit molecules"):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
        else:
            self.log("Converting SMILES to RDKit molecules...")
            mols = [Chem.MolFromSmiles(s) for s in self.smiles]
            mols = [m for m in mols if m is not None]  # Filter out None values
            
        self.log(f"Successfully processed {len(mols)} valid molecules")
        
        # Generate fingerprints using modern RDKit generators
        # We'll generate both NumPy arrays and RDKit ExplicitBitVect objects
        if self.fp_type.lower() == 'morgan':
            # Use modern MorganGenerator
            mgen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.fp_params.get('radius', 2),
                fpSize=self.fp_params.get('nBits', 2048)
            )
            
            if TQDM_AVAILABLE:
                # Generate both formats simultaneously
                fps_rdkit = [mgen.GetFingerprint(mol) for mol in tqdm(mols, desc="Generating Morgan fingerprints")]
                self.fingerprints_rdkit = fps_rdkit
                self.fingerprints = np.array([np.array(fp) for fp in fps_rdkit])
            else:
                self.log("Generating Morgan fingerprints...")
                fps_rdkit = [mgen.GetFingerprint(mol) for mol in mols]
                self.fingerprints_rdkit = fps_rdkit
                self.fingerprints = np.array([np.array(fp) for fp in fps_rdkit])
                
        elif self.fp_type.lower() == 'rdkit':
            # Use modern RDKit fingerprint generator
            rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=self.fp_params.get('fpSize', 2048)
            )
            
            if TQDM_AVAILABLE:
                # Generate both formats simultaneously
                fps_rdkit = [rdgen.GetFingerprint(mol) for mol in tqdm(mols, desc="Generating RDKit fingerprints")]
                self.fingerprints_rdkit = fps_rdkit
                self.fingerprints = np.array([np.array(fp) for fp in fps_rdkit])
            else:
                self.log("Generating RDKit fingerprints...")
                fps_rdkit = [rdgen.GetFingerprint(mol) for mol in mols]
                self.fingerprints_rdkit = fps_rdkit
                self.fingerprints = np.array([np.array(fp) for fp in fps_rdkit])
        else:
            raise ValueError(f"Unsupported fingerprint type: {self.fp_type}")
            
        self.log(f"Generated {len(self.fingerprints)} fingerprints in both formats")
        
        # Save to cache
        self.log(f"Saving NumPy fingerprints to {self.fps_path}")
        np.save(self.fps_path, self.fingerprints)
        
        self.log(f"Saving RDKit fingerprints to {self.fps_rdkit_path}")
        with open(self.fps_rdkit_path, 'wb') as f:
            pickle.dump(self.fingerprints_rdkit, f)
        
        self.log(f"Saving SMILES to {self.smiles_path}")
        with open(self.smiles_path, 'w') as f:
            json.dump(self.smiles, f)
            
        return self.fingerprints, self.smiles
    
    def select_queries(self, n_queries: int = 100, seed: int = 42, force_reload: bool = False) -> List[int]:
        """Select random molecules to use as queries."""
        # Check if we can load from cache
        if not force_reload and os.path.exists(self.queries_path):
            self.log(f"Loading cached query indices from {self.queries_path}")
            with open(self.queries_path, 'r') as f:
                data = json.load(f)
                self.query_indices = data['query_indices']
                
            if self.fingerprints is not None:
                self.query_fps = self.fingerprints[self.query_indices]
            if self.fingerprints_rdkit is not None:
                self.query_fps_rdkit = [self.fingerprints_rdkit[i] for i in self.query_indices]
                
            self.log(f"Loaded {len(self.query_indices)} query indices from cache")
            return self.query_indices
            
        # Make sure fingerprints are loaded
        if self.fingerprints is None:
            self.load_data()
            
        # Select random queries
        self.log(f"Selecting {n_queries} random query molecules with seed {seed}")
        np.random.seed(seed)
        n_mols = len(self.fingerprints)
        self.query_indices = np.random.choice(n_mols, size=min(n_queries, n_mols), replace=False).tolist()
        self.query_fps = self.fingerprints[self.query_indices]
        self.query_fps_rdkit = [self.fingerprints_rdkit[i] for i in self.query_indices]
        
        # Save to cache
        self.log(f"Saving query indices to {self.queries_path}")
        with open(self.queries_path, 'w') as f:
            json.dump({
                'query_indices': self.query_indices,
                'seed': seed,
                'n_queries': len(self.query_indices)
            }, f)
            
        return self.query_indices
    
    def compute_ground_truth(
        self, 
        k_values: List[int] = [10, 50, 100], 
        threshold: float = 0.0,
        force_reload: bool = False
    ) -> Dict:
        """Compute ground truth results using exhaustive search."""
        # Check if we can load from cache
        if not force_reload and os.path.exists(self.ground_truth_path):
            self.log(f"Loading cached ground truth from {self.ground_truth_path}")
            with open(self.ground_truth_path, 'rb') as f:
                self.ground_truth = pickle.load(f)
            
            self.log(f"Loaded ground truth for k values: {list(self.ground_truth.keys())}")
            
            # Check if we have all requested k values
            missing_k = [k for k in k_values if k not in self.ground_truth]
            if not missing_k:
                return self.ground_truth
            else:
                self.log(f"Missing ground truth for k values: {missing_k}")
                # Only compute missing k values
                k_values = missing_k
        
        # Make sure fingerprints and queries are loaded
        if self.fingerprints is None:
            self.load_data()
            
        if self.query_indices is None:
            self.select_queries()
            
        self.log("Computing ground truth using exhaustive search")
        rdkit_engine = RDKitEngine()
        
        # Initialize ground truth if needed
        if self.ground_truth is None:
            self.ground_truth = {}
        
        for k in k_values:
            self.log(f"Computing ground truth for k={k}")
            self.ground_truth[k] = {}
            
            # Process each query with progress reporting
            if TQDM_AVAILABLE:
                for i, query_idx in enumerate(tqdm(self.query_indices, desc=f"Ground truth for k={k}")):
                    query_fp_rdkit = self.fingerprints_rdkit[query_idx]
                    
                    # Get top-k results using exhaustive search with RDKit fingerprints
                    results = rdkit_engine.top_k_similarity(
                        query_fp_rdkit, 
                        self.fingerprints_rdkit, 
                        k=k, 
                        threshold=threshold
                    )
                    
                    self.ground_truth[k][query_idx] = results
            else:
                total_queries = len(self.query_indices)
                for i, query_idx in enumerate(self.query_indices):
                    if i % 10 == 0 or i == total_queries - 1:
                        self.log(f"Processing query {i+1}/{total_queries} for k={k}")
                        
                    query_fp_rdkit = self.fingerprints_rdkit[query_idx]
                    
                    # Get top-k results using exhaustive search with RDKit fingerprints
                    results = rdkit_engine.top_k_similarity(
                        query_fp_rdkit, 
                        self.fingerprints_rdkit, 
                        k=k, 
                        threshold=threshold
                    )
                    
                    self.ground_truth[k][query_idx] = results
            
        # Save to cache
        self.log(f"Saving ground truth to {self.ground_truth_path}")
        with open(self.ground_truth_path, 'wb') as f:
            pickle.dump(self.ground_truth, f)
            
        return self.ground_truth
    
    def build_ivf_index(
        self,
        method: str = 'rdkit',
        n_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
        force_rebuild: bool = False
    ) -> IVFIndex:
        """Build IVF index with BitBIRCH clustering."""
        # Check if we can load from cache
        if not force_rebuild and os.path.exists(self.ivf_index_path):
            self.log(f"Loading cached IVF index from {self.ivf_index_path}")
            with open(self.ivf_index_path, 'rb') as f:
                ivf_index = pickle.load(f)
                
            # Check if the loaded index matches our parameters
            if (ivf_index.similarity_method == method and 
                (n_clusters is None or ivf_index.n_clusters == n_clusters) and
                (threshold is None or abs(ivf_index.threshold - (threshold or self.threshold)) < 1e-6)):
                self.log(f"Loaded IVF index with {ivf_index.n_clusters} clusters")
                return ivf_index
            else:
                self.log(f"Cached index parameters don't match, rebuilding")
        
        # Make sure fingerprints are loaded
        if self.fingerprints is None:
            self.load_data()
            
        # Use default threshold if not specified
        if threshold is None:
            threshold = self.threshold
            
        # Build index
        self.log(f"Building IVF index with BitBIRCH clustering (method={method}, threshold={threshold})")
        ivf_index = IVFIndex(n_clusters=n_clusters, similarity_method=method, threshold=threshold)
        
        build_start_time = time.time()
        ivf_index.build_index(self.fingerprints, self.smiles, self.fingerprints_rdkit)
        build_end_time = time.time()
        
        build_time = build_end_time - build_start_time
        self.log(f"Built IVF index with {ivf_index.n_clusters} clusters in {build_time:.2f} seconds")
        
        # Save to cache
        self.log(f"Saving IVF index to {self.ivf_index_path}")
        with open(self.ivf_index_path, 'wb') as f:
            pickle.dump(ivf_index, f)
            
        return ivf_index
    
    def run_flat_search_benchmark(
        self, 
        method: str = 'rdkit', 
        k_values: List[int] = [10, 50, 100],
        threshold: float = 0.0,
        n_runs: int = 3
    ) -> Dict:
        """Benchmark flat search using specified method."""
        self.log(f"Starting flat search benchmark with {method}")
        
        # Make sure fingerprints and queries are loaded
        if self.fingerprints is None:
            self.load_data()
            
        if self.query_indices is None:
            self.select_queries()
            
        if not self.ground_truth:
            self.compute_ground_truth(k_values=k_values, threshold=threshold)
            
        results = {}
        
        # Initialize engine
        if method.lower() == 'rdkit':
            engine = RDKitEngine()
        elif method.lower() == 'fpsim2':
            engine = FPSim2Engine()
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        for k in k_values:
            self.log(f"Benchmarking flat search with {method} for k={k}")
            results[k] = {
                'query_times': [],
                'recalls': [],
                'results': {}
            }
            
            # Process each query with progress reporting
            if TQDM_AVAILABLE:
                for i, query_idx in enumerate(tqdm(self.query_indices, desc=f"Flat {method} search for k={k}")):
                    # Use appropriate fingerprint format based on method
                    if method.lower() == 'rdkit':
                        query_fp = self.fingerprints_rdkit[query_idx]
                        target_fps = self.fingerprints_rdkit
                    else:
                        query_fp = self.fingerprints[query_idx]
                        target_fps = self.fingerprints
                        
                    gt_results = self.ground_truth[k][query_idx]
                    gt_indices = set(result['index'] for result in gt_results)
                    
                    query_times = []
                    search_results = []
                    
                    # Run benchmark n_runs times
                    for run in range(n_runs):
                        start_time = time.time()
                        run_results = engine.top_k_similarity(
                            query_fp, 
                            target_fps, 
                            k=k, 
                            threshold=threshold
                        )
                        end_time = time.time()
                        
                        query_time = end_time - start_time
                        query_times.append(query_time)
                        search_results.append(run_results)
                        
                    # Use results from last run for recall calculation
                    result_indices = set(result['index'] for result in search_results[-1])
                    recall = len(gt_indices.intersection(result_indices)) / len(gt_indices) if gt_indices else 1.0
                    
                    results[k]['query_times'].extend(query_times)
                    results[k]['recalls'].append(recall)
                    results[k]['results'][str(query_idx)] = search_results[-1]
            else:
                total_queries = len(self.query_indices)
                for i, query_idx in enumerate(self.query_indices):
                    if i % 10 == 0 or i == total_queries - 1:
                        self.log(f"Processing query {i+1}/{total_queries} for k={k}")
                        
                    # Use appropriate fingerprint format based on method
                    if method.lower() == 'rdkit':
                        query_fp = self.fingerprints_rdkit[query_idx]
                        target_fps = self.fingerprints_rdkit
                    else:
                        query_fp = self.fingerprints[query_idx]
                        target_fps = self.fingerprints
                        
                    gt_results = self.ground_truth[k][query_idx]
                    gt_indices = set(result['index'] for result in gt_results)
                    
                    query_times = []
                    search_results = []
                    
                    # Run benchmark n_runs times
                    for run in range(n_runs):
                        start_time = time.time()
                        run_results = engine.top_k_similarity(
                            query_fp, 
                            target_fps, 
                            k=k, 
                            threshold=threshold
                        )
                        end_time = time.time()
                        
                        query_time = end_time - start_time
                        query_times.append(query_time)
                        search_results.append(run_results)
                        
                    # Use results from last run for recall calculation
                    result_indices = set(result['index'] for result in search_results[-1])
                    recall = len(gt_indices.intersection(result_indices)) / len(gt_indices) if gt_indices else 1.0
                    
                    results[k]['query_times'].extend(query_times)
                    results[k]['recalls'].append(recall)
                    results[k]['results'][str(query_idx)] = search_results[-1]
                
            # Calculate average metrics
            results[k]['avg_query_time'] = float(np.mean(results[k]['query_times']))
            results[k]['avg_recall'] = float(np.mean(results[k]['recalls']))
            results[k]['qps'] = float(1.0 / results[k]['avg_query_time'])
            
            self.log(f"k={k}: Avg query time={results[k]['avg_query_time']:.4f}s, "
                  f"QPS={results[k]['qps']:.2f}, Recall={results[k]['avg_recall']:.4f}")
        
        # Convert numpy arrays to lists for JSON serialization
        for k in results:
            results[k]['query_times'] = [float(t) for t in results[k]['query_times']]
            results[k]['recalls'] = [float(r) for r in results[k]['recalls']]
            
        return results
    
    def run_ivf_search_benchmark(
        self, 
        method: str = 'rdkit', 
        k_values: List[int] = [10, 50, 100],
        n_probe_values: List[int] = [1, 2, 4, 8, 16],
        threshold: float = 0.0,
        n_runs: int = 3,
        n_clusters: Optional[int] = None
    ) -> Dict:
        """Benchmark IVF search using BitBIRCH clustering."""
        self.log(f"Starting IVF search benchmark with BitBIRCH+{method}")
        
        # Make sure fingerprints and queries are loaded
        if self.fingerprints is None:
            self.load_data()
            
        if self.query_indices is None:
            self.select_queries()
            
        if not self.ground_truth:
            self.compute_ground_truth(k_values=k_values, threshold=threshold)
            
        results = {}
        
        # Build IVF index
        ivf_index = self.build_ivf_index(
            method=method,
            n_clusters=n_clusters,
            threshold=self.threshold
        )
        
        for k in k_values:
            results[k] = {}
            
            for n_probe in n_probe_values:
                self.log(f"Benchmarking IVF search with {method} for k={k}, n_probe={n_probe}")
                results[k][n_probe] = {
                    'query_times': [],
                    'recalls': [],
                    'results': {}
                }
                
                # Process each query with progress reporting
                if TQDM_AVAILABLE:
                    for i, query_idx in enumerate(tqdm(self.query_indices, 
                                                      desc=f"IVF {method} search for k={k}, n_probe={n_probe}")):
                        # Use appropriate fingerprint format based on method
                        if method.lower() == 'rdkit':
                            query_fp = self.fingerprints_rdkit[query_idx]
                        else:
                            query_fp = self.fingerprints[query_idx]
                            
                        gt_results = self.ground_truth[k][query_idx]
                        gt_indices = set(result['index'] for result in gt_results)
                        
                        query_times = []
                        search_results = []
                        
                        # Run benchmark n_runs times
                        for run in range(n_runs):
                            start_time = time.time()
                            run_results = ivf_index.search(query_fp, k=k, n_probe=n_probe, threshold=threshold)
                            end_time = time.time()
                            
                            query_time = end_time - start_time
                            query_times.append(query_time)
                            search_results.append(run_results)
                            
                        # Use results from last run for recall calculation
                        result_indices = set(result['index'] for result in search_results[-1])
                        recall = len(gt_indices.intersection(result_indices)) / len(gt_indices) if gt_indices else 1.0
                        
                        results[k][n_probe]['query_times'].extend(query_times)
                        results[k][n_probe]['recalls'].append(recall)
                        results[k][n_probe]['results'][str(query_idx)] = search_results[-1]
                else:
                    total_queries = len(self.query_indices)
                    for i, query_idx in enumerate(self.query_indices):
                        if i % 10 == 0 or i == total_queries - 1:
                            self.log(f"Processing query {i+1}/{total_queries} for k={k}, n_probe={n_probe}")
                            
                        # Use appropriate fingerprint format based on method
                        if method.lower() == 'rdkit':
                            query_fp = self.fingerprints_rdkit[query_idx]
                        else:
                            query_fp = self.fingerprints[query_idx]
                            
                        gt_results = self.ground_truth[k][query_idx]
                        gt_indices = set(result['index'] for result in gt_results)
                        
                        query_times = []
                        search_results = []
                        
                        # Run benchmark n_runs times
                        for run in range(n_runs):
                            start_time = time.time()
                            run_results = ivf_index.search(query_fp, k=k, n_probe=n_probe, threshold=threshold)
                            end_time = time.time()
                            
                            query_time = end_time - start_time
                            query_times.append(query_time)
                            search_results.append(run_results)
                            
                        # Use results from last run for recall calculation
                        result_indices = set(result['index'] for result in search_results[-1])
                        recall = len(gt_indices.intersection(result_indices)) / len(gt_indices) if gt_indices else 1.0
                        
                        results[k][n_probe]['query_times'].extend(query_times)
                        results[k][n_probe]['recalls'].append(recall)
                        results[k][n_probe]['results'][str(query_idx)] = search_results[-1]
                        
                # Calculate average metrics
                results[k][n_probe]['avg_query_time'] = float(np.mean(results[k][n_probe]['query_times']))
                results[k][n_probe]['avg_recall'] = float(np.mean(results[k][n_probe]['recalls']))
                results[k][n_probe]['qps'] = float(1.0 / results[k][n_probe]['avg_query_time'])
                
                self.log(f"k={k}, n_probe={n_probe}: Avg query time={results[k][n_probe]['avg_query_time']:.4f}s, "
                      f"QPS={results[k][n_probe]['qps']:.2f}, Recall={results[k][n_probe]['avg_recall']:.4f}")
        
        # Convert numpy arrays to lists for JSON serialization
        for k in results:
            for n_probe in results[k]:
                results[k][n_probe]['query_times'] = [float(t) for t in results[k][n_probe]['query_times']]
                results[k][n_probe]['recalls'] = [float(r) for r in results[k][n_probe]['recalls']]
            
        return results
    
    def generate_plots(self, flat_results: Dict, ivf_results: Dict) -> None:
        """Generate benchmark plots."""
        self.log("Generating benchmark plots")
        
        # Plot 1: Query time vs k for different methods
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Flat search results
        k_values = sorted(flat_results.keys())
        flat_times = [flat_results[k]['avg_query_time'] for k in k_values]
        flat_recalls = [flat_results[k]['avg_recall'] for k in k_values]
        
        ax1.plot(k_values, flat_times, 'o-', label='Flat Search', linewidth=2, markersize=8)
        ax1.set_xlabel('k (top-k results)')
        ax1.set_ylabel('Average Query Time (s)')
        ax1.set_title('Query Time vs k')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(k_values, flat_recalls, 'o-', label='Flat Search', linewidth=2, markersize=8)
        ax2.set_xlabel('k (top-k results)')
        ax2.set_ylabel('Average Recall')
        ax2.set_title('Recall vs k')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # IVF search results for different n_probe values
        if ivf_results:
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, n_probe in enumerate(sorted(ivf_results[k_values[0]].keys())):
                ivf_times = [ivf_results[k][n_probe]['avg_query_time'] for k in k_values]
                ivf_recalls = [ivf_results[k][n_probe]['avg_recall'] for k in k_values]
                
                color = colors[i % len(colors)]
                ax1.plot(k_values, ivf_times, 'o--', label=f'IVF (n_probe={n_probe})', 
                        color=color, linewidth=2, markersize=6)
                ax2.plot(k_values, ivf_recalls, 'o--', label=f'IVF (n_probe={n_probe})', 
                        color=color, linewidth=2, markersize=6)
            
            ax1.legend()
            ax2.legend()
            
            # Plot 3: Recall vs Query Time trade-off
            flat_time = flat_results[k_values[-1]]['avg_query_time']  # Use largest k for comparison
            flat_recall = flat_results[k_values[-1]]['avg_recall']
            
            ax3.scatter([flat_time], [flat_recall], s=100, c='blue', marker='o', label='Flat Search')
            
            for i, n_probe in enumerate(sorted(ivf_results[k_values[-1]].keys())):
                ivf_time = ivf_results[k_values[-1]][n_probe]['avg_query_time']
                ivf_recall = ivf_results[k_values[-1]][n_probe]['avg_recall']
                color = colors[i % len(colors)]
                ax3.scatter([ivf_time], [ivf_recall], s=100, c=color, marker='s', 
                           label=f'IVF (n_probe={n_probe})')
            
            ax3.set_xlabel('Average Query Time (s)')
            ax3.set_ylabel('Average Recall')
            ax3.set_title(f'Recall vs Query Time Trade-off (k={k_values[-1]})')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Plot 4: Speedup vs Recall trade-off
            for i, n_probe in enumerate(sorted(ivf_results[k_values[-1]].keys())):
                ivf_time = ivf_results[k_values[-1]][n_probe]['avg_query_time']
                ivf_recall = ivf_results[k_values[-1]][n_probe]['avg_recall']
                speedup = flat_time / ivf_time if ivf_time > 0 else 0
                color = colors[i % len(colors)]
                ax4.scatter([speedup], [ivf_recall], s=100, c=color, marker='s', 
                           label=f'IVF (n_probe={n_probe})')
            
            ax4.axvline(x=1.0, color='blue', linestyle='--', alpha=0.5, label='No speedup')
            ax4.set_xlabel('Speedup vs Flat Search')
            ax4.set_ylabel('Average Recall')  
            ax4.set_title(f'Speedup vs Recall Trade-off (k={k_values[-1]})')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'benchmark_results_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Plots saved to {plot_path}")
    
    def save_results(self, flat_results: Dict, ivf_results: Dict) -> None:
        """Save benchmark results to JSON file."""
        self.log(f"Saving results to {self.results_path}")
        
        results = {
            'dataset': self.dataset_name,
            'dataset_path': self.dataset_path,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'fp_type': self.fp_type,
                'fp_size': self.fp_size,
                'radius': self.radius,
                'threshold': self.threshold
            },
            'data_size': len(self.smiles),
            'n_queries': len(self.query_indices) if self.query_indices else 0,
            'flat_search_results': flat_results,
            'ivf_search_results': ivf_results
        }
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
        self.log(f"Results saved to {self.results_path}")
    
    def run_full_benchmark(
        self,
        methods: List[str] = ['rdkit'],
        k_values: List[int] = [10, 50, 100],
        n_probe_values: List[int] = [1, 2, 4, 8, 16],
        threshold: float = 0.0,
        n_runs: int = 3,
        n_queries: int = 100,
        n_clusters: Optional[int] = None,
        stages: Optional[List[str]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Run the complete benchmark pipeline.
        
        Args:
            methods: List of similarity search methods to test
            k_values: List of k values for top-k search
            n_probe_values: List of n_probe values for IVF search
            threshold: Similarity threshold
            n_runs: Number of runs per query for averaging
            n_queries: Number of query molecules
            n_clusters: Number of clusters for IVF index
            stages: List of stages to run ('data', 'queries', 'ground_truth', 'flat', 'ivf', 'plots')
        
        Returns:
            Tuple of (flat_results, ivf_results)
        """
        # Default stages
        if stages is None:
            stages = ['data', 'queries', 'ground_truth', 'flat', 'ivf', 'plots']
        
        self.log(f"Running benchmark pipeline with stages: {stages}")
        
        flat_results = {}
        ivf_results = {}
        
        # Stage 1: Load data and generate fingerprints
        if 'data' in stages:
            self.log("=== Stage 1: Loading data and generating fingerprints ===")
            self.load_data()
        
        # Stage 2: Select query molecules
        if 'queries' in stages:
            self.log("=== Stage 2: Selecting query molecules ===")
            self.select_queries(n_queries=n_queries)
        
        # Stage 3: Compute ground truth
        if 'ground_truth' in stages:
            self.log("=== Stage 3: Computing ground truth ===")
            self.compute_ground_truth(k_values=k_values, threshold=threshold)
        
        # Stage 4: Run flat search benchmarks
        if 'flat' in stages:
            self.log("=== Stage 4: Running flat search benchmarks ===")
            for method in methods:
                self.log(f"Running flat search benchmark for {method}")
                method_results = self.run_flat_search_benchmark(
                    method=method,
                    k_values=k_values,
                    threshold=threshold,
                    n_runs=n_runs
                )
                flat_results[method] = method_results
        
        # Stage 5: Run IVF search benchmarks
        if 'ivf' in stages:
            self.log("=== Stage 5: Running IVF search benchmarks ===")
            for method in methods:
                self.log(f"Running IVF search benchmark for {method}")
                method_results = self.run_ivf_search_benchmark(
                    method=method,
                    k_values=k_values,
                    n_probe_values=n_probe_values,
                    threshold=threshold,
                    n_runs=n_runs,
                    n_clusters=n_clusters
                )
                ivf_results[method] = method_results
        
        # Stage 6: Generate plots and save results
        if 'plots' in stages:
            self.log("=== Stage 6: Generating plots and saving results ===")
            if flat_results and ivf_results:
                # Use first method's results for plotting
                first_method = list(flat_results.keys())[0]
                self.generate_plots(flat_results[first_method], ivf_results[first_method])
            
            # Save all results
            self.save_results(flat_results, ivf_results)
        
        self.log("=== Benchmark completed ===")
        return flat_results, ivf_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run improved BitBIRCH IVF search benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark on ChEMBL data
  python improved_benchmark.py --dataset data/chembl_33_np.smi
  
  # Run only specific stages
  python improved_benchmark.py --dataset data/chembl_33_np.smi --stages data queries ground_truth
  
  # Run with custom parameters
  python improved_benchmark.py --dataset data/chembl_33_np.smi --fp-size 1024 --n-queries 50
  
  # Resume from a specific stage (reusing cached files)
  python improved_benchmark.py --dataset data/chembl_33_np.smi --stages ivf plots
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True, help='Path to SMILES dataset file')
    
    # Optional arguments
    parser.add_argument('--output-dir', default='search/results', help='Output directory for results')
    parser.add_argument('--fp-type', default='morgan', choices=['morgan', 'rdkit'], 
                       help='Fingerprint type')
    parser.add_argument('--fp-size', type=int, default=2048, help='Fingerprint size in bits')
    parser.add_argument('--radius', type=int, default=2, help='Morgan fingerprint radius')
    parser.add_argument('--threshold', type=float, default=0.65, help='Similarity threshold')
    parser.add_argument('--methods', nargs='+', default=['rdkit'], choices=['rdkit', 'fpsim2'],
                       help='Similarity search methods to test')
    parser.add_argument('--k-values', nargs='+', type=int, default=[10, 50, 100],
                       help='k values for top-k search')
    parser.add_argument('--n-probe-values', nargs='+', type=int, default=[1, 2, 4, 8, 16],
                       help='n_probe values for IVF search')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of runs per query for averaging')
    parser.add_argument('--n-queries', type=int, default=100, help='Number of query molecules')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters for IVF index')
    parser.add_argument('--stages', nargs='*', 
                       choices=['data', 'queries', 'ground_truth', 'flat', 'ivf', 'plots'],
                       help='Specific stages to run (default: all)')
    parser.add_argument('--force-reload', action='store_true', 
                       help='Force reload of cached files')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    verbose = args.verbose and not args.quiet
    
    # Create benchmark instance
    benchmark = StagedBenchmark(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        fp_type=args.fp_type,
        fp_size=args.fp_size,
        radius=args.radius,
        threshold=args.threshold,
        verbose=verbose
    )
    
    # Run benchmark
    try:
        flat_results, ivf_results = benchmark.run_full_benchmark(
            methods=args.methods,
            k_values=args.k_values,
            n_probe_values=args.n_probe_values,
            threshold=args.threshold,
            n_runs=args.n_runs,
            n_queries=args.n_queries,
            n_clusters=args.n_clusters,
            stages=args.stages
        )
        
        # Print summary
        print("\n=== BENCHMARK SUMMARY ===")
        print(f"Dataset: {args.dataset}")
        print(f"Fingerprint: {args.fp_type} ({args.fp_size} bits)")
        print(f"Data size: {len(benchmark.smiles)}")
        print(f"Queries: {len(benchmark.query_indices) if benchmark.query_indices else 0}")
        
        if flat_results:
            print("\nFlat Search Results:")
            for method, results in flat_results.items():
                print(f"  {method.upper()}:")
                for k in sorted(results.keys()):
                    print(f"    k={k}: {results[k]['qps']:.1f} QPS, "
                          f"Recall={results[k]['avg_recall']:.3f}")
        
        if ivf_results:
            print("\nIVF Search Results:")
            for method, results in ivf_results.items():
                print(f"  {method.upper()}:")
                for k in sorted(results.keys()):
                    print(f"    k={k}:")
                    for n_probe in sorted(results[k].keys()):
                        print(f"      n_probe={n_probe}: {results[k][n_probe]['qps']:.1f} QPS, "
                              f"Recall={results[k][n_probe]['avg_recall']:.3f}")
        
        print(f"\nResults saved to: {benchmark.results_path}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())