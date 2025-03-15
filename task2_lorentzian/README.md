# Task 2: Lorentzian Resonance Behavior

## Problem Statement

Analyze the resonance behavior of a dissipative, driven harmonic oscillator in both time and frequency domains. The system is described by:

Time domain: 
d²x/dt² + γ(dx/dt) + ω₀²x = Fe^(iωₑt)

Frequency domain: 
-ω²x̃ + iγωx̃ + ω₀²x̃ = Fδ(ω-ωₑ)

Show that the energy absorption per cycle follows a Lorentzian distribution:

E = (Fπγωₑ)/((ω₀² - ωₑ²)² + γ²ωₑ²)

## Solution Approach

The solution involves:

1. Solving the equation of motion in the frequency domain to find the complex amplitude
2. Calculating the energy absorption per cycle using the imaginary part of the response
3. Simplifying the expression to obtain the Lorentzian form
4. Visualizing the Lorentzian resonance curves for different damping parameters

## Mathematical Derivation

1. The complex amplitude in frequency domain is:
   x̃(ω) = F/(-ω² + iγω + ω₀²)

2. At the driving frequency ωₑ, the amplitude is:
   A = F/(-ωₑ² + iγωₑ + ω₀²)

3. The energy absorption per cycle is proportional to the imaginary part:
   E = Fπ·Im(A)·ωₑ

4. After simplification, this yields the Lorentzian form:
   E = (Fπγωₑ)/((ω₀² - ωₑ²)² + γ²ωₑ²)

## Implementation

The implementation includes:

1. Symbolic derivation using SymPy to verify the mathematical result
2. Numerical calculation of the Lorentzian function
3. Visualization of resonance curves for different damping coefficients

## How to Run

```bash
python lorentzian_resonance.py
```

## Output

The script generates:
- A step-by-step symbolic derivation of the Lorentzian formula
- A plot showing Lorentzian resonance curves for different damping values
- The plot is saved as `lorentzian_resonance.png`

## Key Features of Lorentzian Resonance

- Maximum energy absorption occurs near (but not exactly at) the natural frequency ω₀
- Lower damping (γ) creates sharper, higher peaks
- The width of the peak is proportional to the damping coefficient
- The total area under each curve is π/2, independent of γ 