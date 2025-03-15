#!/usr/bin/env python3
"""
Test script for the Heisenberg XXX Hamiltonian implementation
"""

from hamiltonian import HeisenbergXXX
import numpy as np

def test_hamiltonian():
    """Test the Heisenberg XXX Hamiltonian implementation"""
    print("Testing Heisenberg XXX Hamiltonian implementation...")
    
    # Create a model with 3 spins
    model = HeisenbergXXX(n_spins=3, J=1.0)
    
    # Check basis states
    print(f"Number of basis states: {len(model.basis_states)}")
    print(f"Basis states: {model.basis_states}")
    
    # Check Hamiltonian dimensions
    H = model.hamiltonian
    print(f"Hamiltonian dimensions: {H.shape}")
    
    # Check if Hamiltonian is Hermitian
    is_hermitian = np.allclose(H, H.T)
    print(f"Hamiltonian is Hermitian: {is_hermitian}")
    
    # Check eigenvalues
    eigenvalues, _ = np.linalg.eigh(H)
    print(f"Eigenvalues: {eigenvalues}")
    
    # Check magnon states
    magnon_states, magnon_energies = model.get_magnon_states()
    print(f"Magnon states: {magnon_states}")
    print(f"Magnon energies: {magnon_energies}")
    
    print("Hamiltonian test completed successfully!")

if __name__ == "__main__":
    test_hamiltonian() 