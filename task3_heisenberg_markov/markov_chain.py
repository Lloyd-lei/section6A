import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
from hamiltonian import HeisenbergXXX

# Set plot style for better aesthetics
plt.style.use('dark_background')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = (10, 6)

class MarkovChain:
    """
    Markov Chain analysis for the Heisenberg XXX model
    """
    
    def __init__(self, heisenberg_model, temperature=1.0):
        """
        Initialize the Markov Chain
        
        Parameters:
        -----------
        heisenberg_model : HeisenbergXXX
            Heisenberg XXX model instance
        temperature : float
            Temperature parameter (kB*T)
        """
        self.model = heisenberg_model
        self.temperature = temperature
        self.basis_states = heisenberg_model.basis_states
        self.n_states = len(self.basis_states)
        
        # Construct transition matrices
        self.site_transition_matrix = self._construct_site_transition_matrix()
        
        # Get magnon states and energies
        self.magnon_states, self.magnon_energies = self.model.get_magnon_states()
        self.n_magnons = len(self.magnon_states)
        
        # Construct magnon transition matrix
        self.magnon_transition_matrix = self._construct_magnon_transition_matrix()
    
    def _construct_site_transition_matrix(self):
        """
        Construct the transition matrix in the site basis
        
        Returns:
        --------
        P : numpy.ndarray
            Transition matrix
        """
        # Get the Hamiltonian matrix
        H = self.model.hamiltonian
        
        # Calculate energy differences for transitions
        # For detailed balance: P_ij / P_ji = exp(-(E_j - E_i)/kT)
        P = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j and abs(H[i, j]) > 1e-10:  # Non-zero off-diagonal element
                    energy_diff = H[j, j] - H[i, i]  # Energy difference
                    # Metropolis-like transition probability
                    P[i, j] = abs(H[i, j]) * np.exp(-max(0, energy_diff) / self.temperature)
        
        # Ensure rows sum to 1 (stochastic matrix)
        row_sums = np.sum(P, axis=1)
        for i in range(self.n_states):
            if row_sums[i] > 0:
                P[i, :] /= row_sums[i]
            else:
                # If no transitions, stay in the same state
                P[i, i] = 1.0
        
        # Ensure P is a valid stochastic matrix (no negative values)
        P = np.abs(P)
        
        # Re-normalize
        row_sums = np.sum(P, axis=1)
        for i in range(self.n_states):
            if row_sums[i] > 0:
                P[i, :] /= row_sums[i]
        
        return P
    
    def _construct_magnon_transition_matrix(self):
        """
        Construct the transition matrix in the magnon basis
        
        Returns:
        --------
        P : numpy.ndarray
            Transition matrix for magnons
        """
        # Initialize transition matrix
        P = np.zeros((self.n_magnons, self.n_magnons))
        
        # Calculate transition probabilities based on Boltzmann factors
        for i in range(self.n_magnons):
            for j in range(self.n_magnons):
                if i != j:
                    energy_diff = self.magnon_energies[j] - self.magnon_energies[i]
                    P[i, j] = np.exp(-max(0, energy_diff) / self.temperature)
        
        # Ensure rows sum to 1 (stochastic matrix)
        row_sums = np.sum(P, axis=1)
        for i in range(self.n_magnons):
            if row_sums[i] > 0:
                P[i, :] /= row_sums[i]
            else:
                # If no transitions, stay in the same state
                P[i, i] = 1.0
        
        return P
    
    def find_stationary_distribution_site(self):
        """
        Find the stationary distribution for the site basis Markov chain
        
        Returns:
        --------
        pi : numpy.ndarray
            Stationary distribution
        """
        # Method 1: Using eigenvalue decomposition
        # The stationary distribution is the left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.site_transition_matrix.T)
        
        # Find the index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        # Get the corresponding eigenvector
        pi = np.real(eigenvectors[:, idx])
        
        # Take absolute values to ensure no negative probabilities
        pi = np.abs(pi)
        
        # Normalize to ensure sum = 1
        pi = pi / np.sum(pi)
        
        return pi
    
    def find_stationary_distribution_magnon(self):
        """
        Find the stationary distribution for the magnon basis Markov chain
        
        Returns:
        --------
        pi : numpy.ndarray
            Stationary distribution
        """
        # Method 1: Using eigenvalue decomposition
        # The stationary distribution is the left eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.magnon_transition_matrix.T)
        
        # Find the index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        
        # Get the corresponding eigenvector
        pi = np.real(eigenvectors[:, idx])
        
        # Take absolute values to ensure no negative probabilities
        pi = np.abs(pi)
        
        # Normalize to ensure sum = 1
        pi = pi / np.sum(pi)
        
        return pi
    
    def power_iteration_site(self, initial_dist, n_iterations=100):
        """
        Perform power iteration to find the stationary distribution in site basis
        
        Parameters:
        -----------
        initial_dist : numpy.ndarray
            Initial distribution
        n_iterations : int
            Number of iterations
            
        Returns:
        --------
        pi_history : numpy.ndarray
            History of distributions during iteration
        """
        pi = initial_dist.copy()
        pi_history = np.zeros((n_iterations + 1, self.n_states))
        pi_history[0] = pi
        
        for i in range(n_iterations):
            pi = pi @ self.site_transition_matrix
            pi_history[i + 1] = pi
        
        return pi_history
    
    def power_iteration_magnon(self, initial_dist, n_iterations=100):
        """
        Perform power iteration to find the stationary distribution in magnon basis
        
        Parameters:
        -----------
        initial_dist : numpy.ndarray
            Initial distribution
        n_iterations : int
            Number of iterations
            
        Returns:
        --------
        pi_history : numpy.ndarray
            History of distributions during iteration
        """
        pi = initial_dist.copy()
        pi_history = np.zeros((n_iterations + 1, self.n_magnons))
        pi_history[0] = pi
        
        for i in range(n_iterations):
            pi = pi @ self.magnon_transition_matrix
            pi_history[i + 1] = pi
        
        return pi_history
    
    def transition_matrix_to_rate_matrix(self, P, delta_t=0.1):
        """
        Convert transition matrix to rate matrix
        
        Parameters:
        -----------
        P : numpy.ndarray
            Transition matrix
        delta_t : float
            Time step
            
        Returns:
        --------
        Q : numpy.ndarray
            Rate matrix
        """
        # Q ≈ (1/Δt) * (P - I)
        # For more accuracy: Q ≈ (1/Δt) * log(P)
        try:
            Q = logm(P) / delta_t
        except:
            # Fallback to approximation if matrix logarithm fails
            Q = (P - np.eye(P.shape[0])) / delta_t
        
        return Q
    
    def solve_master_equation(self, initial_dist, t_span, basis='magnon'):
        """
        Solve the master equation dπ/dt = πQ
        
        Parameters:
        -----------
        initial_dist : numpy.ndarray
            Initial distribution
        t_span : tuple
            Time span (t_start, t_end)
        basis : str
            'site' or 'magnon'
            
        Returns:
        --------
        t : numpy.ndarray
            Time points
        pi_history : numpy.ndarray
            History of distributions
        """
        if basis == 'site':
            P = self.site_transition_matrix
            n_states = self.n_states
        else:
            P = self.magnon_transition_matrix
            n_states = self.n_magnons
        
        # Convert transition matrix to rate matrix
        Q = self.transition_matrix_to_rate_matrix(P)
        
        # Ensure Q is real
        Q = np.real(Q)
        
        # Define the ODE system
        def master_equation(t, pi):
            return pi @ Q
        
        # Solve the ODE
        solution = solve_ivp(
            master_equation, 
            t_span, 
            initial_dist, 
            method='RK45',
            t_eval=np.linspace(t_span[0], t_span[1], 100)
        )
        
        return solution.t, solution.y.T
    
    def plot_stationary_distributions(self):
        """
        Plot the stationary distributions for both site and magnon bases
        """
        # Get stationary distributions
        pi_site = self.find_stationary_distribution_site()
        pi_magnon = self.find_stationary_distribution_magnon()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot site basis stationary distribution
        ax1.bar(range(self.n_states), pi_site, color='cyan', alpha=0.7)
        ax1.set_xlabel('State Index', fontsize=14)
        ax1.set_ylabel('Probability', fontsize=14)
        ax1.set_title(f'Stationary Distribution in Site Basis (T={self.temperature})', fontsize=16)
        ax1.set_xticks(range(self.n_states))
        ax1.set_xticklabels(self.basis_states, rotation=45)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot magnon basis stationary distribution
        ax2.bar(range(self.n_magnons), pi_magnon, color='magenta', alpha=0.7)
        ax2.set_xlabel('Magnon State', fontsize=14)
        ax2.set_ylabel('Probability', fontsize=14)
        ax2.set_title(f'Stationary Distribution in Magnon Basis (T={self.temperature})', fontsize=16)
        ax2.set_xticks(range(self.n_magnons))
        ax2.set_xticklabels(self.magnon_states)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add explanation text
        site_props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        site_explanation = (
            "Site basis: Probability of each spin configuration\n"
            f"Temperature: T = {self.temperature}\n"
            "Higher probability for lower energy states at low T"
        )
        ax1.text(0.02, 0.95, site_explanation, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=site_props)
        
        magnon_props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        magnon_explanation = (
            "Magnon basis: Probability of each magnon state\n"
            f"Temperature: T = {self.temperature}\n"
            "Magnons are collective excitations (spin waves)"
        )
        ax2.text(0.02, 0.95, magnon_explanation, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=magnon_props)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'task3_heisenberg_markov/stationary_distributions_T{self.temperature}.png', 
                   dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
    
    def plot_power_iteration_convergence(self, initial_dists, labels, basis='site'):
        """
        Plot the convergence of power iteration for different initial distributions
        
        Parameters:
        -----------
        initial_dists : list
            List of initial distributions
        labels : list
            Labels for the initial distributions
        basis : str
            'site' or 'magnon'
        """
        # Number of iterations
        n_iterations = 50
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for different initial distributions
        colors = ['cyan', 'magenta', 'yellow']
        
        # Get stationary distribution
        if basis == 'site':
            pi_stationary = self.find_stationary_distribution_site()
            n_states = self.n_states
        else:
            pi_stationary = self.find_stationary_distribution_magnon()
            n_states = self.n_magnons
        
        # Plot convergence for each initial distribution
        for i, (initial_dist, label) in enumerate(zip(initial_dists, labels)):
            # Perform power iteration
            if basis == 'site':
                pi_history = self.power_iteration_site(initial_dist, n_iterations)
            else:
                pi_history = self.power_iteration_magnon(initial_dist, n_iterations)
            
            # Calculate distance to stationary distribution
            distances = np.zeros(n_iterations + 1)
            for j in range(n_iterations + 1):
                distances[j] = np.linalg.norm(pi_history[j] - pi_stationary)
            
            # Plot distance
            ax.plot(range(n_iterations + 1), distances, color=colors[i], linewidth=2.5,
                   label=f'Initial: {label}')
        
        # Add labels and title
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Distance to Stationary Distribution', fontsize=14)
        ax.set_title(f'Convergence of Power Iteration in {basis.capitalize()} Basis (T={self.temperature})', 
                    fontsize=16)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        
        # Set log scale for y-axis
        ax.set_yscale('log')
        
        # Add explanation text
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        explanation = (
            f"Power iteration: π_{{{n_iterations+1}}} = π_{{{n_iterations}}}P\n"
            "Convergence rate depends on the second largest eigenvalue of P\n"
            f"Temperature: T = {self.temperature}"
        )
        ax.text(0.02, 0.95, explanation, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'task3_heisenberg_markov/power_iteration_{basis}_T{self.temperature}.png', 
                   dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
    
    def plot_master_equation_evolution(self, initial_dist, t_span, basis='magnon'):
        """
        Plot the evolution of the master equation
        
        Parameters:
        -----------
        initial_dist : numpy.ndarray
            Initial distribution
        t_span : tuple
            Time span (t_start, t_end)
        basis : str
            'site' or 'magnon'
        """
        # Solve the master equation
        t, pi_history = self.solve_master_equation(initial_dist, t_span, basis)
        
        # Get state labels
        if basis == 'site':
            states = self.basis_states
        else:
            states = self.magnon_states
        
        n_states = len(states)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for different states
        cmap = plt.cm.viridis
        colors = [cmap(i / n_states) for i in range(n_states)]
        
        # Plot evolution for each state
        for i in range(n_states):
            ax.plot(t, pi_history[:, i], color=colors[i], linewidth=2.5,
                   label=f'State: {states[i]}')
        
        # Add labels and title
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.set_title(f'Master Equation Evolution in {basis.capitalize()} Basis (T={self.temperature})', 
                    fontsize=16)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12)
        
        # Add explanation text
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        explanation = (
            "Master equation: dπ/dt = πQ\n"
            "Q is the transition rate matrix\n"
            f"Temperature: T = {self.temperature}"
        )
        ax.text(0.02, 0.95, explanation, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'task3_heisenberg_markov/master_equation_{basis}_T{self.temperature}.png', 
                   dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()

def main():
    """
    Main function to analyze the Heisenberg XXX model using Markov chains
    """
    # Create a Heisenberg XXX model with 3 spins
    model = HeisenbergXXX(n_spins=3, J=1.0)
    
    # Create Markov chain for different temperatures
    temperatures = [0.1, 1.0, 10.0]
    
    for temp in temperatures:
        print(f"\n--- Temperature T = {temp} ---")
        markov = MarkovChain(model, temperature=temp)
        
        # Question 1 & 2: Find stationary distribution in site basis
        pi_site = markov.find_stationary_distribution_site()
        print("\nStationary Distribution in Site Basis:")
        for i, state in enumerate(model.basis_states):
            print(f"{state}: {pi_site[i]:.4f}")
        
        # Question 3: Power iteration in site basis
        print("\nPower Iteration in Site Basis:")
        
        # Initial distribution 1: All probability on |↑↑↑⟩
        initial_dist1 = np.zeros(markov.n_states)
        initial_dist1[model.basis_states.index('↑↑↑')] = 1.0
        
        # Initial distribution 2: Equal probability on |↑↑↑⟩ and |↓↑↓⟩
        initial_dist2 = np.zeros(markov.n_states)
        initial_dist2[model.basis_states.index('↑↑↑')] = 0.5
        initial_dist2[model.basis_states.index('↓↑↓')] = 0.5
        
        # Initial distribution 3: Uniform distribution
        initial_dist3 = np.ones(markov.n_states) / markov.n_states
        
        # Plot convergence
        markov.plot_power_iteration_convergence(
            [initial_dist1, initial_dist2, initial_dist3],
            ['|↑↑↑⟩', '|↑↑↑⟩ & |↓↑↓⟩', 'Uniform'],
            basis='site'
        )
        
        # Question 4 & 5: Find stationary distribution in magnon basis
        pi_magnon = markov.find_stationary_distribution_magnon()
        print("\nStationary Distribution in Magnon Basis:")
        for i, state in enumerate(markov.magnon_states):
            print(f"{state}: {pi_magnon[i]:.4f}")
        
        # Question 6: Power iteration in magnon basis
        print("\nPower Iteration in Magnon Basis:")
        
        # Initial distribution 1: All probability on |k=1⟩
        initial_dist1 = np.zeros(markov.n_magnons)
        initial_dist1[1] = 1.0
        
        # Initial distribution 2: Equal probability on |k=1⟩ and |k=4⟩
        # Note: For N=3, we only have k=0,1,2, so we'll use k=2 instead of k=4
        initial_dist2 = np.zeros(markov.n_magnons)
        initial_dist2[1] = 0.5
        initial_dist2[2 % markov.n_magnons] = 0.5
        
        # Initial distribution 3: Uniform distribution
        initial_dist3 = np.ones(markov.n_magnons) / markov.n_magnons
        
        # Plot convergence
        markov.plot_power_iteration_convergence(
            [initial_dist1, initial_dist2, initial_dist3],
            ['|k=1⟩', '|k=1⟩ & |k=2⟩', 'Uniform'],
            basis='magnon'
        )
        
        # Plot stationary distributions
        markov.plot_stationary_distributions()
        
        # Question 7: Master equation evolution
        print("\nMaster Equation Evolution:")
        
        # Initial distribution: All probability on |k=1⟩
        initial_dist = np.zeros(markov.n_magnons)
        initial_dist[1] = 1.0
        
        # Solve and plot
        markov.plot_master_equation_evolution(initial_dist, (0, 10), basis='magnon')

if __name__ == "__main__":
    main() 