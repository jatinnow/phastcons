# The Interactive Mathematical PhastCons Lab

A comprehensive educational Streamlit application for understanding the PhastCons algorithm through interactive mathematical exploration.

## Overview

This application implements the complete PhastCons algorithm (Siepel et al., 2005) with 8 interactive modules that expose the mathematical "engine" of phylogenetic Hidden Markov Models for conservation detection.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run phastcons_lab.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## Features

### Module 1: Continuous-Time Markov Chain (The Foundation)
- Interactive rate matrix exploration
- Jukes-Cantor and HKY85 evolutionary models
- Real-time computation of P(t) = exp(Qt)
- Visual heatmaps of transition probabilities

### Module 2: Likelihood of Single Column
- Felsenstein's pruning algorithm implementation
- Interactive phylogenetic tree ((A:0.1, B:0.1):0.1, C:0.2)
- Partial likelihood visualization
- Understanding ancestral state marginalization

### Module 3: Two Models (Rate Scaling)
- Conserved vs. Neutral model comparison
- Log-likelihood ratio computation
- Interactive ρ (rho) parameter for conservation strength
- Visual comparison of model preferences

### Module 4: HMM Structure
- Hidden state transitions (Neutral ↔ Conserved)
- Expected conserved element length calculation
- Stationary distribution computation
- Transition matrix visualization

### Module 5: Forward-Backward Algorithm
- Complete sequence likelihood calculation
- Forward (α) table visualization
- Real-time probability propagation
- Dynamic sequence input

### Module 6: EM Algorithm (Learning)
- Expectation-Maximization parameter estimation
- Posterior probability computation
- Interactive single-step EM iteration
- Before/after parameter comparison

### Module 7: KL Divergence & Phylogenetic Information Threshold
- KL divergence computation
- Minimum detectable length (L_min) calculation
- Conservation detection power analysis
- Interactive visualization of detection threshold

### Module 8: Viterbi Decoding
- Most likely state path discovery
- Conserved element annotation
- Viterbi matrix visualization
- Path backtracking demonstration

## Educational Goals

This tool is designed so that students can:
1. Understand the mathematical foundations of PhastCons
2. See how parameters affect calculations in real-time
3. Gain intuition for phylogenetic HMMs
4. Re-derive the PhastCons paper from first principles

## Technical Implementation

### Core Classes

- **ContinuousTimeMarkovChain**: Handles evolutionary models and matrix exponentials
- **PhylogeneticTree**: Implements Felsenstein's pruning algorithm
- **PhyloHMM**: Complete phylogenetic HMM with Forward-Backward and Viterbi

### Key Algorithms

- Matrix exponential computation via `scipy.linalg.expm`
- Dynamic programming for Forward-Backward
- Viterbi algorithm for optimal path finding
- EM algorithm for parameter estimation

## Usage Tips

1. **Start with Module 1**: Build intuition for continuous-time evolution
2. **Progress Sequentially**: Each module builds on previous concepts
3. **Experiment with Parameters**: Use sliders to see real-time updates
4. **Compare Results**: Try different sequences and parameter combinations
5. **Read the Math**: LaTeX equations explain what's being computed

## Example Workflow

1. **Module 1**: Set α = 1.0, t = 0.1, observe P(t)
2. **Module 2**: Input "A, A, A" - see high likelihood
3. **Module 3**: Set ρ = 0.3, compare AAA vs. ACG
4. **Module 4**: Adjust μ and ν, observe element length changes
5. **Module 5**: Input "AAAGGTTT", view forward probabilities
6. **Module 6**: Run EM to see parameter learning
7. **Module 7**: Explore detection thresholds at different ρ values
8. **Module 8**: Decode "AAAAGGGGGTTTT" to find conserved regions

## Mathematical Reference

The application implements concepts from:

**Siepel, A., Bejerano, G., Pedersen, J. S., Hinrichs, A. S., Hou, M., Rosenbloom, K., ... & Haussler, D. (2005).** 
*Evolutionarily conserved elements in vertebrate, insect, worm, and yeast genomes.* 
Genome research, 15(8), 1034-1050.

## Troubleshooting

### Common Issues

**"Invalid sequence!" error**: 
- Use only A, C, G, T nucleotides
- Check for spaces or invalid characters

**Slow performance**: 
- Long sequences (>20 bp) may take time
- Forward-Backward is O(L·S²) where L = length, S = states

**Display issues**: 
- Refresh the browser
- Check console for errors
- Ensure all dependencies are installed

## Advanced Features

### Modifying the Tree
Edit the `PhylogeneticTree` class to change:
- Tree topology
- Branch lengths
- Number of species

### Adding New Models
Extend `ContinuousTimeMarkovChain` with:
- GTR (General Time Reversible)
- Custom rate matrices
- Different equilibrium frequencies

### Custom Visualizations
Use matplotlib/seaborn to add:
- ROC curves for detection
- Parameter sensitivity plots
- Convergence trajectories

## License

Educational use - based on published scientific algorithms.

## Contributing

Suggestions for improvements:
- Additional evolutionary models
- More complex tree structures
- Performance optimizations
- Additional visualizations

## Contact

For questions about the implementation or educational use, refer to the original PhastCons paper and documentation.

---

**Built with**: Streamlit, NumPy, SciPy, Pandas, Matplotlib, Seaborn

**Tested on**: Python 3.8+, Modern web browsers (Chrome, Firefox, Safari)
