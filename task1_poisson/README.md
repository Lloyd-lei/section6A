# Task 1: Poisson Distribution for Random Star Distribution

## Problem Statement

Consider an example in astrophysics: If stars are randomly distributed around us with density n, what is the probability that the nearest star is at distance R?

## Solution Approach

This problem can be solved using Poisson statistics. The key insight is that for a random (Poisson) distribution of stars:

1. The probability of finding exactly k stars in a volume V is given by the Poisson distribution:
   P(k, V) = (nV)^k * e^(-nV) / k!

2. For the nearest star problem, we need to find the probability that:
   - There are 0 stars within radius r
   - AND there is at least 1 star in the spherical shell between r and r+dr

3. The probability of finding no stars within radius r is:
   P(0, V) = e^(-nV) where V = (4π/3)r³

4. The probability density function (PDF) for the nearest star being at distance r is:
   P(r) = d/dr(1-e^(-(4π/3)r³n)) = 4πr²n * e^(-(4π/3)r³n)

## Implementation

The implementation includes:

1. Theoretical calculation of the PDF
2. Monte Carlo simulation to verify the theoretical result
3. Visualization comparing theory and simulation

## How to Run

```bash
python poisson_stars.py
```

## Output

The script generates a plot showing:
- Theoretical probability density function
- Histogram of simulated nearest star distances
- KDE (Kernel Density Estimation) of the simulation results

The plot is saved as `nearest_star_distribution.png`. 