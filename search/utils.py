"""
Utility functions for the IVF search implementation.

This module provides helper functions for loading data, converting
between different fingerprint formats, and other common operations.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any

def load_smiles_file(file_path: str) -> List[str]:
    """
    Load SMILES from a file.
    
    Args:
        file_path: Path to SMILES file
        
    Returns:
        List of SMILES strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r') as f:
        # Handle different SMILES file formats
        lines = f.readlines()
        smiles = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract SMILES from the line
            parts = line.split()
            if parts:
                smiles.append(parts[0])  # Assume first column is SMILES
                
    return smiles

def generate_fingerprints(
    smiles: List[str], 
    fp_type: str = 'morgan', 
    fp_params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Generate fingerprints from SMILES.
    
    Args:
        smiles: List of SMILES strings
        fp_type: Type of fingerprint ('morgan', 'rdkit', etc.)
        fp_params: Parameters for fingerprint generation
        
    Returns:
        NumPy array of fingerprints
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    if fp_params is None:
        if fp_type.lower() == 'morgan':
            fp_params = {'radius': 2, 'nBits': 2048}
        elif fp_type.lower() == 'rdkit':
            fp_params = {'fpSize': 2048}
        else:
            fp_params = {}
    
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]  # Filter out None values
    
    # Generate fingerprints
    fingerprints = []
    
    if fp_type.lower() == 'morgan':
        fingerprints = np.array([
            np.array(AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                fp_params.get('radius', 2), 
                nBits=fp_params.get('nBits', 2048)
            )) 
            for mol in mols
        ])
    elif fp_type.lower() == 'rdkit':
        fingerprints = np.array([
            np.array(Chem.RDKFingerprint(
                mol,
                fpSize=fp_params.get('fpSize', 2048)
            ))
            for mol in mols
        ])
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
        
    return fingerprints

def load_fingerprints(
    file_path: str,
    fp_type: str = 'morgan',
    fp_params: Dict[str, Any] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load SMILES and generate fingerprints.
    
    Args:
        file_path: Path to SMILES file
        fp_type: Type of fingerprint ('morgan', 'rdkit', etc.)
        fp_params: Parameters for fingerprint generation
        
    Returns:
        Tuple of (fingerprints, smiles)
    """
    smiles = load_smiles_file(file_path)
    fingerprints = generate_fingerprints(smiles, fp_type, fp_params)
    
    return fingerprints, smiles

def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Calculate Tanimoto similarity between two fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Tanimoto similarity score
    """
    # Check if fingerprints are binary
    if not (set(np.unique(fp1)).issubset({0, 1}) and set(np.unique(fp2)).issubset({0, 1})):
        raise ValueError("Fingerprints must be binary (0 or 1)")
        
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    
    if union == 0:
        return 1.0  # Both fingerprints are all zeros
        
    return intersection / union

def bulk_tanimoto_similarity(query_fp: np.ndarray, target_fps: np.ndarray) -> np.ndarray:
    """
    Calculate Tanimoto similarities between a query and multiple targets.
    
    Args:
        query_fp: Query fingerprint
        target_fps: Target fingerprints
        
    Returns:
        Array of Tanimoto similarity scores
    """
    # This is a simple implementation, prefer using RDKit's BulkTanimotoSimilarity
    # for better performance in production
    return np.array([tanimoto_similarity(query_fp, fp) for fp in target_fps])

def create_fpsim2_db(
    smiles_file: str,
    output_file: str,
    fp_type: str = 'Morgan',
    fp_params: Dict[str, Any] = None
) -> str:
    """
    Create FPSim2 database from SMILES file.
    
    Args:
        smiles_file: Path to SMILES file
        output_file: Path to output database file
        fp_type: Fingerprint type ('Morgan', 'RDKit', etc.)
        fp_params: Parameters for fingerprint generation
        
    Returns:
        Path to created database file
    """
    try:
        from FPSim2.io import create_db_file
    except ImportError:
        raise ImportError("FPSim2 not installed. Install with 'pip install FPSim2'")
    
    if fp_params is None:
        if fp_type == 'Morgan':
            fp_params = {'radius': 2, 'fpSize': 2048}
        else:
            fp_params = {'fpSize': 2048}
    
    create_db_file(
        mols_source=smiles_file,
        filename=output_file,
        mol_format=None,  # Auto-detect
        fp_type=fp_type,
        fp_params=fp_params
    )
    
    return output_file