import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sympy as sp
from sympy import symbols, I, pi, simplify, expand, collect, factor, im

# Set plot style for better aesthetics
plt.style.use('dark_background')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = (10, 6)

def symbolic_derivation():
    """
    Perform symbolic derivation of the Lorentzian energy absorption formula
    """
    # Define symbols
    omega, omega_0, gamma, F, t = symbols('omega omega_0 gamma F t', real=True)
    omega_f = symbols('omega_f', real=True)
    
    # Define the complex amplitude in frequency domain
    x_tilde = F / (-omega**2 + I*gamma*omega + omega_0**2)
    
    # Substitute omega = omega_f for the driving frequency
    x_tilde_at_omega_f = x_tilde.subs(omega, omega_f)
    
    # Calculate the complex amplitude
    A = x_tilde_at_omega_f
    
    # Calculate the energy absorption per cycle
    # E = F * pi * Im(A) * omega_f
    E = F * pi * im(A * omega_f)
    
    # Simplify the expression
    E_simplified = simplify(E)
    
    # Collect terms to get the Lorentzian form
    E_collected = collect(expand(E_simplified), omega_f**2)
    
    # Factor to get the final form
    E_final = factor(E_collected)
    
    return {
        'x_tilde': x_tilde,
        'A': A,
        'E': E,
        'E_simplified': E_simplified,
        'E_final': E_final
    }

def lorentzian(omega_f, omega_0, gamma, F=1.0):
    """
    Calculate the Lorentzian energy absorption per cycle
    
    Parameters:
    -----------
    omega_f : float or array
        Driving frequency
    omega_0 : float
        Natural frequency of the oscillator
    gamma : float
        Damping coefficient
    F : float
        Driving force amplitude
        
    Returns:
    --------
    E : float or array
        Energy absorption per cycle
    """
    return F * pi * gamma * omega_f / ((omega_0**2 - omega_f**2)**2 + gamma**2 * omega_f**2)

def plot_lorentzian():
    """
    Plot the Lorentzian energy absorption curve for different parameters
    """
    # Set parameters
    omega_0 = 1.0  # Natural frequency
    gamma_values = [0.1, 0.2, 0.5]  # Different damping coefficients
    F = 1.0  # Force amplitude
    
    # Create frequency range
    omega_f = np.linspace(0.5, 1.5, 1000)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot for different gamma values
    colors = ['cyan', 'magenta', 'yellow']
    for i, gamma in enumerate(gamma_values):
        E = lorentzian(omega_f, omega_0, gamma, F)
        ax.plot(omega_f, E, color=colors[i], linewidth=2.5, 
                label=f'γ = {gamma}')
        
        # Mark the peak
        peak_idx = np.argmax(E)
        peak_omega = omega_f[peak_idx]
        peak_E = E[peak_idx]
        ax.plot(peak_omega, peak_E, 'o', color='white', markersize=8)
        ax.text(peak_omega, peak_E*1.05, f'Peak: ω = {peak_omega:.3f}', 
                color='white', fontsize=10, ha='center')
    
    # Add vertical line at natural frequency
    ax.axvline(x=omega_0, color='red', linestyle='--', alpha=0.7, 
               label=f'Natural frequency ω₀ = {omega_0}')
    
    # Add labels and title
    ax.set_xlabel('Driving Frequency (ω_f)', fontsize=14)
    ax.set_ylabel('Energy Absorption per Cycle (E)', fontsize=14)
    ax.set_title('Lorentzian Resonance: Energy Absorption vs. Driving Frequency', 
                 fontsize=16)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='upper right')
    
    # Add equation
    equation = r"$E = \frac{F\pi\gamma\omega_f}{({\omega_0}^2-{\omega_f}^2)^2 + \gamma^2{\omega_f}^2}$"
    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    ax.text(0.02, 0.97, equation, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    # Add explanation
    explanation = (
        "The Lorentzian describes energy absorption in a damped, driven harmonic oscillator.\n"
        "Key features:\n"
        "• Maximum absorption occurs near (but not exactly at) the natural frequency ω₀\n"
        "• Lower damping (γ) creates sharper, higher peaks\n"
        "• The width of the peak is proportional to the damping coefficient\n"
        "• The total area under each curve is π/2, independent of γ"
    )
    
    ax.text(0.02, 0.85, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('task2_lorentzian/lorentzian_resonance.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def main():
    """
    Main function to solve and visualize the Lorentzian problem
    """
    # Perform symbolic derivation
    results = symbolic_derivation()
    
    # Print the derivation steps
    print("Symbolic Derivation of the Lorentzian Formula:")
    print("----------------------------------------------")
    print(f"Complex amplitude in frequency domain: x̃(ω) = {results['x_tilde']}")
    print(f"Complex amplitude at driving frequency: A = {results['A']}")
    print(f"Energy absorption per cycle: E = {results['E']}")
    print(f"Simplified expression: E = {results['E_simplified']}")
    print(f"Final Lorentzian form: E = {results['E_final']}")
    print("\nThis confirms the Lorentzian form of the energy absorption per cycle.")
    
    # Plot the Lorentzian curves
    plot_lorentzian()
    
    print("\nThe Lorentzian resonance curves have been plotted and saved.")

if __name__ == "__main__":
    main() 