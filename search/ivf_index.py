"""
IVF (Inverted File) search index implementation using BitBIRCH clustering.

This module provides an efficient search index for chemical fingerprints
by utilizing BitBIRCH clustering to partition the search space and enable
fast approximate nearest neighbor search.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Union, Optional

# Import BitBIRCH for clustering
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitbirch.bitbirch import BitBirch, set_merge


class IVFIndex:
    """
    Inverted File (IVF) index for efficient similarity search of chemical fingerprints.
    
    The index uses BitBIRCH clustering to partition fingerprints into clusters,
    then at query time, only the most relevant clusters are searched, providing
    a significant speedup over exhaustive search.
    
    Attributes:
        n_clusters (int): Number of clusters to use. If None, uses sqrt(n_samples)
        similarity_method (str): Method to use for similarity calculations ('rdkit' or 'fpsim2')
        threshold (float): Similarity threshold for BitBIRCH clustering
        branching_factor (int): Branching factor for BitBIRCH clustering
        cluster_centroids (np.ndarray): Centroids of each cluster
        cluster_members (Dict[int, List[int]]): Mapping of cluster IDs to member fingerprint indices
        fingerprints (np.ndarray): Stored fingerprints for similarity search
        smiles (List[str]): Optional SMILES strings corresponding to fingerprints
        built (bool): Whether the index has been built
    """
    
    def __init__(
        self, 
        n_clusters: Optional[int] = None, 
        similarity_method: str = 'rdkit',
        threshold: float = 0.7, 
        branching_factor: int = 50
    ):
        """
        Initialize the IVF index.
        
        Args:
            n_clusters: Number of clusters to use. If None, uses sqrt(n_samples)
            similarity_method: Method for similarity calculations ('rdkit' or 'fpsim2')
            threshold: Similarity threshold for BitBIRCH clustering
            branching_factor: Branching factor for BitBIRCH clustering
        """
        self.n_clusters = n_clusters
        self.similarity_method = similarity_method.lower()
        self.threshold = threshold
        self.branching_factor = branching_factor
        
        # Will be populated during build_index
        self.cluster_centroids = None
        self.cluster_centroids_rdkit = None  # RDKit format for performance
        self.cluster_members = {}
        self.fingerprints = None
        self.fingerprints_rdkit = None  # RDKit format for performance
        self.smiles = None
        self.built = False
        
        # Validate similarity method
        if self.similarity_method not in ['rdkit', 'fpsim2']:
            raise ValueError("similarity_method must be 'rdkit' or 'fpsim2'")
        
    def build_index(self, fingerprints: np.ndarray, smiles: Optional[List[str]] = None, fingerprints_rdkit: Optional[List] = None) -> None:
        """
        Build the IVF index by clustering fingerprints using BitBIRCH.
        
        Args:
            fingerprints: Binary fingerprints of shape (n_samples, n_features)
            smiles: Optional list of SMILES strings corresponding to fingerprints
            fingerprints_rdkit: Optional list of RDKit ExplicitBitVect objects for performance
        """
        n_samples = fingerprints.shape[0]
        
        # Store fingerprints and smiles for later use
        self.fingerprints = fingerprints
        self.fingerprints_rdkit = fingerprints_rdkit
        self.smiles = smiles
        
        # Determine number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = int(math.sqrt(n_samples))
            print(f"Using {self.n_clusters} clusters (sqrt of {n_samples} samples)")
            
        # Initialize BitBIRCH for clustering
        set_merge('diameter')  # Using diameter merge criterion
        birch = BitBirch(threshold=self.threshold, branching_factor=self.branching_factor)
        
        # Fit BitBIRCH to fingerprints
        print(f"Clustering {n_samples} fingerprints into {self.n_clusters} clusters...")
        birch.fit(fingerprints)
        
        # Extract cluster information
        cluster_ids = birch.get_assignments(n_mols=n_samples)
        unique_clusters = np.unique(cluster_ids)
        
        # Calculate centroids and organize members
        print(f"Found {len(unique_clusters)} unique clusters")
        self.cluster_centroids = []
        self.cluster_members = {}
        
        # Get centroids and member indices for each cluster
        self.cluster_centroids_rdkit = []
        for cluster_id in unique_clusters:
            # Get indices of fingerprints in this cluster
            member_indices = np.where(cluster_ids == cluster_id)[0]
            self.cluster_members[cluster_id] = member_indices
            
            # Get fingerprints in this cluster
            cluster_fps = fingerprints[member_indices]
            
            # Calculate centroid (use actual centroid or pick medoid)
            # For now, use first fingerprint as representative (will be improved)
            self.cluster_centroids.append(cluster_fps[0])
            
            # Also store RDKit format if available
            if self.fingerprints_rdkit is not None:
                representative_idx = member_indices[0]
                self.cluster_centroids_rdkit.append(self.fingerprints_rdkit[representative_idx])
            
        # Convert centroids to numpy array
        self.cluster_centroids = np.array(self.cluster_centroids)
        
        self.built = True
        print(f"IVF index built with {len(self.cluster_centroids)} clusters")
        
    def _find_nearest_clusters(self, query_fp, n_probe: int) -> List[int]:
        """
        Find the n_probe nearest clusters to the query fingerprint.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            n_probe: Number of clusters to return
            
        Returns:
            List of cluster IDs, sorted by similarity to query
        """
        from rdkit import DataStructs
        import numpy as np
        
        # Use RDKit fingerprints directly if available
        if (hasattr(self, 'cluster_centroids_rdkit') and 
            self.cluster_centroids_rdkit and 
            isinstance(query_fp, DataStructs.ExplicitBitVect)):
            # Direct RDKit calculation - no conversion needed!
            similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, self.cluster_centroids_rdkit))
        else:
            # Fall back to conversion if needed
            if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                for i, val in enumerate(query_fp):
                    if val == 1:
                        query_bitvect.SetBit(i)
            else:
                query_bitvect = query_fp
                
            # Convert centroids to ExplicitBitVect list if needed
            centroid_bitvects = []
            for centroid in self.cluster_centroids:
                if not isinstance(centroid, DataStructs.ExplicitBitVect):
                    bitvect = DataStructs.ExplicitBitVect(len(centroid))
                    for i, val in enumerate(centroid):
                        if val == 1:
                            bitvect.SetBit(i)
                    centroid_bitvects.append(bitvect)
                else:
                    centroid_bitvects.append(centroid)
            
            # Calculate similarities to all centroids
            similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, centroid_bitvects))
        
        # Get indices of top n_probe most similar centroids
        top_indices = np.argsort(similarities)[-n_probe:][::-1]  # Sort descending
        
        # Map index to cluster ID
        cluster_ids = list(self.cluster_members.keys())
        nearest_clusters = [cluster_ids[idx] for idx in top_indices]
        
        return nearest_clusters
    
    def search(
        self, 
        query_fp, 
        k: int = 10, 
        n_probe: int = 1,
        threshold: float = 0.0
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        Search for the k most similar fingerprints to the query.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            k: Number of results to return
            n_probe: Number of clusters to search
            threshold: Minimum similarity threshold (0.0 means no threshold)
            
        Returns:
            List of dictionaries containing search results, each with:
                - 'index': Index of the fingerprint
                - 'similarity': Tanimoto similarity to query
                - 'smiles': SMILES string (if available)
        """
        if not self.built:
            raise RuntimeError("Index has not been built. Call build_index first.")
            
        # Find nearest clusters
        nearest_clusters = self._find_nearest_clusters(query_fp, n_probe)
        
        # Get indices of fingerprints in the selected clusters
        candidate_indices = []
        for cluster_id in nearest_clusters:
            candidate_indices.extend(self.cluster_members[cluster_id])
            
        # Calculate similarities based on method and available formats
        if self.similarity_method == 'rdkit':
            from rdkit import DataStructs
            
            # Use RDKit fingerprints directly if available
            if (hasattr(self, 'fingerprints_rdkit') and 
                self.fingerprints_rdkit and 
                isinstance(query_fp, DataStructs.ExplicitBitVect)):
                # Direct RDKit calculation - no conversion needed!
                candidate_rdkit_fps = [self.fingerprints_rdkit[i] for i in candidate_indices]
                similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, candidate_rdkit_fps))
            else:
                # Fall back to conversion if needed
                if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                    query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                    for i, val in enumerate(query_fp):
                        if val == 1:
                            query_bitvect.SetBit(i)
                else:
                    query_bitvect = query_fp
                    
                # Get candidate fingerprints and convert if needed
                candidate_fps = self.fingerprints[candidate_indices]
                candidate_bitvects = []
                for fp in candidate_fps:
                    if not isinstance(fp, DataStructs.ExplicitBitVect):
                        bitvect = DataStructs.ExplicitBitVect(len(fp))
                        for i, val in enumerate(fp):
                            if val == 1:
                                bitvect.SetBit(i)
                        candidate_bitvects.append(bitvect)
                    else:
                        candidate_bitvects.append(fp)
                        
                # Calculate similarities
                similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, candidate_bitvects))
            
        elif self.similarity_method == 'fpsim2':
            # For now, use a placeholder implementation; will be replaced with actual FPSim2
            # In the full implementation, this would use FPSim2's optimized similarity calculation
            from rdkit import DataStructs
            
            # Similar optimization can be applied here for FPSim2
            if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                for i, val in enumerate(query_fp):
                    if val == 1:
                        query_bitvect.SetBit(i)
            else:
                query_bitvect = query_fp
                
            # Get candidate fingerprints and convert if needed
            candidate_fps = self.fingerprints[candidate_indices]
            candidate_bitvects = []
            for fp in candidate_fps:
                if not isinstance(fp, DataStructs.ExplicitBitVect):
                    bitvect = DataStructs.ExplicitBitVect(len(fp))
                    for i, val in enumerate(fp):
                        if val == 1:
                            bitvect.SetBit(i)
                    candidate_bitvects.append(bitvect)
                else:
                    candidate_bitvects.append(fp)
                    
            # Calculate similarities
            similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, candidate_bitvects))
            
        # Apply threshold filter
        if threshold > 0.0:
            valid_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
            similarities = [similarities[i] for i in valid_indices]
            candidate_indices = [candidate_indices[i] for i in valid_indices]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1][:k]
        
        # Prepare results
        results = []
        for idx in sorted_indices:
            result = {
                'index': candidate_indices[idx],
                'similarity': similarities[idx],
            }
            
            # Add SMILES if available
            if self.smiles is not None:
                result['smiles'] = self.smiles[candidate_indices[idx]]
                
            results.append(result)
            
        return results