import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams

# Set plot style for better aesthetics
plt.style.use('dark_background')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = (10, 6)

def theoretical_nearest_star_pdf(r, n):
    """
    Calculate the theoretical probability density function for the distance to the nearest star
    
    Parameters:
    -----------
    r : float or array
        Distance from observer
    n : float
        Number density of stars (stars per unit volume)
        
    Returns:
    --------
    pdf : float or array
        Probability density at distance r
    """
    # The PDF is derived from Poisson statistics
    # P(r) = 4πr²n * exp(-4πr³n/3)
    return 4 * np.pi * r**2 * n * np.exp(-(4/3) * np.pi * r**3 * n)

def simulate_nearest_star_distances(n, num_samples=10000, box_size=100):
    """
    Simulate random star distributions and find distances to nearest stars
    
    Parameters:
    -----------
    n : float
        Number density of stars (stars per unit volume)
    num_samples : int
        Number of simulation samples
    box_size : float
        Size of the simulation box
        
    Returns:
    --------
    distances : array
        Array of distances to nearest stars
    """
    # Calculate expected number of stars in the box
    expected_stars = int(n * box_size**3)
    
    # Array to store nearest distances
    nearest_distances = np.zeros(num_samples)
    
    # Run simulations
    for i in range(num_samples):
        # Generate random stars in the box
        num_stars = np.random.poisson(expected_stars)
        if num_stars == 0:
            # If no stars, set a large distance
            nearest_distances[i] = box_size
            continue
            
        # Generate random positions for stars
        stars = np.random.uniform(0, box_size, size=(num_stars, 3))
        
        # Observer at center of box
        observer = np.array([box_size/2, box_size/2, box_size/2])
        
        # Calculate distances from observer to all stars
        distances = np.sqrt(np.sum((stars - observer)**2, axis=1))
        
        # Find the nearest star
        nearest_distances[i] = np.min(distances)
    
    return nearest_distances

def main():
    """
    Main function to solve and visualize the Poisson problem
    """
    # Set the number density of stars (arbitrary units)
    n = 0.001  # stars per cubic unit
    
    # Theoretical calculation
    r_values = np.linspace(0, 20, 1000)
    pdf_values = theoretical_nearest_star_pdf(r_values, n)
    
    # Simulation
    nearest_distances = simulate_nearest_star_distances(n, num_samples=50000)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot theoretical PDF
    ax.plot(r_values, pdf_values, 'r-', linewidth=2, 
            label='Theoretical PDF: $4\\pi r^2 n \\exp(-\\frac{4\\pi}{3}r^3 n)$')
    
    # Plot histogram of simulated distances
    hist_kwargs = {
        'bins': 50,
        'density': True,
        'alpha': 0.6,
        'color': 'cyan'
    }
    ax.hist(nearest_distances, **hist_kwargs, label='Simulation')
    
    # Add a KDE plot for the simulated data
    kde = stats.gaussian_kde(nearest_distances)
    ax.plot(r_values, kde(r_values), 'g--', linewidth=2, label='KDE of Simulation')
    
    # Add labels and title
    ax.set_xlabel('Distance to Nearest Star (r)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('Probability Distribution of Distance to Nearest Star\n'
                 'in a Random Star Distribution', fontsize=16)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Add explanation text
    explanation = (
        "For randomly distributed stars with density n,\n"
        "the probability that the nearest star is at distance r\n"
        "follows a distribution derived from Poisson statistics.\n\n"
        "The probability of finding no stars within radius r is:\n"
        "$P(0, V) = e^{-nV}$ where $V = \\frac{4\\pi}{3}r^3$\n\n"
        "The PDF is then: $P(r) = \\frac{d}{dr}(1-e^{-\\frac{4\\pi}{3}r^3 n}) = 4\\pi r^2 n e^{-\\frac{4\\pi}{3}r^3 n}$"
    )
    
    # Add text box with explanation
    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    ax.text(0.65, 0.97, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('task1_poisson/nearest_star_distribution.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print("The probability distribution of the nearest star distance has been calculated and plotted.")
    print("The theoretical PDF is: P(r) = 4πr²n * exp(-4πr³n/3)")
    print("This result is derived from Poisson statistics for randomly distributed stars.")

if __name__ == "__main__":
    main() 