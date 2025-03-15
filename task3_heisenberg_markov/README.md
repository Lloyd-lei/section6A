# Task 3: Heisenberg XXX Hamiltonian Markov Chain Analysis

## Problem Statement

Analyze the Heisenberg XXX Hamiltonian on a ring using Markov chain techniques. The Hamiltonian is given by:

H = (J*N/4) - J * Σ[1/2(S⁺ᵢS⁻ᵢ₊₁ + S⁻ᵢS⁺ᵢ₊₁) + SᶻᵢSᶻᵢ₊₁]

with periodic boundary conditions (site N+1 = site 1).

## Implementation

This task is divided into several parts:

1. **Hamiltonian Construction**: Implementation of the Heisenberg XXX Hamiltonian in the site basis
2. **Site Basis Markov Chain**: Construction and analysis of the Markov chain in the site basis
3. **Magnon Basis Markov Chain**: Construction and analysis of the Markov chain in the magnon basis
4. **Master Equation**: Conversion of the discrete-time Markov chain to a continuous-time master equation

## Files

- `hamiltonian.py`: Implementation of the Heisenberg XXX Hamiltonian
- `markov_chain.py`: Implementation of Markov chain analysis for both site and magnon bases

## Questions Addressed

### Site Basis Analysis
1. Construction of the Markov chain transition matrix in the site basis
2. Finding the stationary distribution by solving πP = π
3. Using power iteration to find the stationary distribution for different initial states

### Magnon Basis Analysis
4. Construction of the Markov chain in the magnon basis
5. Finding the stationary distribution in the magnon basis
6. Using power iteration in the magnon basis
7. Converting the transition matrix to a rate matrix and solving the master equation

## Theoretical Background

### Site Basis
The site basis consists of all possible spin configurations (e.g., |↑↑↓⟩). For N=3 spins, there are 2³ = 8 basis states.

### Magnon Basis
Magnons are collective excitations (spin waves) in the system. For a ring with N sites, the magnon states are characterized by momentum k = 2πm/N, where m = 0, 1, ..., N-1.

### Markov Chain
The transition matrix P is constructed such that:
- P_ij represents the probability of transitioning from state i to state j
- The stationary distribution π satisfies πP = π and Σπ_i = 1
- The transition probabilities respect detailed balance: P_ij/P_ji = exp(-(E_j - E_i)/kT)

### Master Equation
The continuous-time master equation is given by:
dπ/dt = πQ
where Q is the rate matrix derived from the transition matrix P.

## How to Run

```bash
python markov_chain.py
```

## Output

The script generates several visualizations:
- Stationary distributions in both site and magnon bases
- Convergence of power iteration for different initial distributions
- Time evolution of the master equation

All plots are saved in the task3_heisenberg_markov directory with descriptive filenames. 