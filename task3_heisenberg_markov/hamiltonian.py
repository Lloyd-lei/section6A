import numpy as np
from itertools import product

class HeisenbergXXX:
    """
    Implementation of the Heisenberg XXX Hamiltonian on a ring
    """
    
    def __init__(self, n_spins=3, J=1.0):
        """
        Initialize the Heisenberg XXX model
        
        Parameters:
        -----------
        n_spins : int
            Number of spins in the chain
        J : float
            Coupling constant (J > 0 for ferromagnetic, J < 0 for antiferromagnetic)
        """
        self.n_spins = n_spins
        self.J = J
        self.hilbert_dim = 2**n_spins
        self.basis_states = self._generate_basis_states()
        self.hamiltonian = self._construct_hamiltonian()
        
    def _generate_basis_states(self):
        """
        Generate all possible basis states in the computational basis
        
        Returns:
        --------
        basis_states : list
            List of all basis states as strings (e.g., '↑↑↓')
        """
        # Generate all possible combinations of up and down spins
        spin_configs = list(product(['↑', '↓'], repeat=self.n_spins))
        
        # Convert tuples to strings
        basis_states = [''.join(config) for config in spin_configs]
        
        return basis_states
    
    def _apply_s_plus(self, state, site):
        """
        Apply S+ operator to a state at a specific site
        
        Parameters:
        -----------
        state : str
            Spin state as a string (e.g., '↑↑↓')
        site : int
            Site index (0-based)
            
        Returns:
        --------
        new_state : str or None
            Resulting state after applying S+, or None if result is zero
        coef : float
            Coefficient of the resulting state
        """
        state_list = list(state)
        
        # S+ |↑⟩ = 0
        if state_list[site] == '↑':
            return None, 0
        
        # S+ |↓⟩ = |↑⟩
        if state_list[site] == '↓':
            state_list[site] = '↑'
            return ''.join(state_list), 1.0
            
        return None, 0
    
    def _apply_s_minus(self, state, site):
        """
        Apply S- operator to a state at a specific site
        
        Parameters:
        -----------
        state : str
            Spin state as a string (e.g., '↑↑↓')
        site : int
            Site index (0-based)
            
        Returns:
        --------
        new_state : str or None
            Resulting state after applying S-, or None if result is zero
        coef : float
            Coefficient of the resulting state
        """
        state_list = list(state)
        
        # S- |↓⟩ = 0
        if state_list[site] == '↓':
            return None, 0
        
        # S- |↑⟩ = |↓⟩
        if state_list[site] == '↑':
            state_list[site] = '↓'
            return ''.join(state_list), 1.0
            
        return None, 0
    
    def _apply_s_z(self, state, site):
        """
        Apply Sz operator to a state at a specific site
        
        Parameters:
        -----------
        state : str
            Spin state as a string (e.g., '↑↑↓')
        site : int
            Site index (0-based)
            
        Returns:
        --------
        new_state : str
            The same state (Sz is diagonal)
        coef : float
            Eigenvalue of Sz (+1/2 for ↑, -1/2 for ↓)
        """
        # Sz |↑⟩ = (1/2) |↑⟩
        if state[site] == '↑':
            return state, 0.5
        
        # Sz |↓⟩ = (-1/2) |↓⟩
        if state[site] == '↓':
            return state, -0.5
            
        return state, 0
    
    def _construct_hamiltonian(self):
        """
        Construct the Hamiltonian matrix for the Heisenberg XXX model
        
        Returns:
        --------
        H : numpy.ndarray
            Hamiltonian matrix
        """
        n = self.hilbert_dim
        H = np.zeros((n, n), dtype=float)
        
        # Constant term: J*N/4
        constant_term = self.J * self.n_spins / 4
        np.fill_diagonal(H, constant_term)
        
        # Iterate over all basis states
        for i, state_i in enumerate(self.basis_states):
            # Iterate over all sites with periodic boundary
            for site in range(self.n_spins):
                next_site = (site + 1) % self.n_spins
                
                # Apply S+_i S-_{i+1} term
                new_state_plus, coef_plus = self._apply_s_plus(state_i, site)
                if new_state_plus is not None and coef_plus != 0:
                    new_state_final, coef_minus = self._apply_s_minus(new_state_plus, next_site)
                    if new_state_final is not None and coef_minus != 0:
                        j = self.basis_states.index(new_state_final)
                        H[i, j] -= self.J * 0.5 * coef_plus * coef_minus
                
                # Apply S-_i S+_{i+1} term
                new_state_minus, coef_minus = self._apply_s_minus(state_i, site)
                if new_state_minus is not None and coef_minus != 0:
                    new_state_final, coef_plus = self._apply_s_plus(new_state_minus, next_site)
                    if new_state_final is not None and coef_plus != 0:
                        j = self.basis_states.index(new_state_final)
                        H[i, j] -= self.J * 0.5 * coef_minus * coef_plus
                
                # Apply Sz_i Sz_{i+1} term
                _, coef_z_i = self._apply_s_z(state_i, site)
                _, coef_z_next = self._apply_s_z(state_i, next_site)
                H[i, i] -= self.J * coef_z_i * coef_z_next
        
        return H
    
    def get_magnon_states(self):
        """
        Generate the magnon states for the system
        
        Returns:
        --------
        magnon_states : list
            List of magnon states with their momenta
        magnon_energies : list
            Corresponding energies of the magnon states
        """
        # For N=3 case, the allowed momenta are k=0,1,2
        k_values = np.arange(self.n_spins)
        
        # Calculate magnon energies
        magnon_energies = 2 * self.J * np.sin(np.pi * k_values / self.n_spins)**2
        
        # Generate magnon states (simplified representation)
        magnon_states = [f"|k={k}⟩" for k in k_values]
        
        return magnon_states, magnon_energies
    
    def print_hamiltonian(self):
        """
        Print the Hamiltonian matrix with basis state labels
        """
        print("Heisenberg XXX Hamiltonian Matrix:")
        print("Basis states:", self.basis_states)
        print(self.hamiltonian)
        
        # Print eigenvalues
        eigenvalues, _ = np.linalg.eigh(self.hamiltonian)
        print("\nEigenvalues:", eigenvalues)

if __name__ == "__main__":
    # Create a Heisenberg XXX model with 3 spins
    model = HeisenbergXXX(n_spins=3, J=1.0)
    
    # Print the Hamiltonian matrix
    model.print_hamiltonian()
    
    # Print magnon states and energies
    magnon_states, magnon_energies = model.get_magnon_states()
    print("\nMagnon States:", magnon_states)
    print("Magnon Energies:", magnon_energies) 