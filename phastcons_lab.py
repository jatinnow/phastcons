"""
The Interactive Mathematical PhastCons Lab
A comprehensive educational tool for understanding PhastCons algorithm

Author: Computational Biology Expert
Built with: Streamlit, NumPy, SciPy, Pandas
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
from typing import Tuple, List, Dict
import itertools

# Set page configuration
st.set_page_config(
    page_title="PhastCons Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CORE PHYLOGENETIC-HMM CLASSES
# ============================================================================

class ContinuousTimeMarkovChain:
    """Handles continuous-time Markov chain evolution"""
    
    def __init__(self, nucleotides=['A', 'C', 'G', 'T']):
        self.nucleotides = nucleotides
        self.n_states = len(nucleotides)
        self.nuc_to_idx = {nuc: i for i, nuc in enumerate(nucleotides)}
    
    def jukes_cantor_rate_matrix(self, alpha=1.0):
        """Jukes-Cantor with Q normalization"""
        Q = np.ones((self.n_states, self.n_states)) * alpha
        np.fill_diagonal(Q, 0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        
        pi = np.ones(self.n_states) / self.n_states
        r = -np.sum(pi * np.diagonal(Q))
        Q = Q / r
        return Q
    
    def hky85_rate_matrix(self, kappa=2.0, pi=None):
        """HKY85 time-reversible: Q[i,j] = rate × π_j"""
        if pi is None:
            pi = np.ones(4) / 4
        
        Q = np.zeros((4, 4))
        
        Q[0, 2] = kappa * pi[2]  # A -> G
        Q[2, 0] = kappa * pi[0]  # G -> A
        Q[1, 3] = kappa * pi[3]
        Q[3, 1] = kappa * pi[1]
        
        Q[0, 1] = pi[1]; Q[0, 3] = pi[3]
        Q[1, 0] = pi[0]; Q[1, 2] = pi[2]
        Q[2, 1] = pi[1]; Q[2, 3] = pi[3]
        Q[3, 0] = pi[0]; Q[3, 2] = pi[2]
        
        np.fill_diagonal(Q, 0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        
        r = -np.sum(pi * np.diagonal(Q))
        Q = Q / r
        return Q
    
    def transition_probability_matrix(self, Q, t):
        return expm(Q * t)
    
    def equilibrium_distribution(self, Q):
        n = Q.shape[0]
        A = np.vstack([Q.T, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.maximum(pi, 0)
        pi = pi / pi.sum()
        return pi




class PhylogeneticTree:
    """Simple phylogenetic tree for likelihood calculations"""
    
    def __init__(self):
        # Simple 3-leaf tree: ((A:0.1, B:0.1):0.1, C:0.2)
        #     root
        #    /    \
        #   n1     C (0.2)
        #  /  \
        # A    B (0.1 each)
        self.structure = {
            'root': {'children': ['n1', 'C'], 'branch_lengths': [0.1, 0.2]},
            'n1': {'children': ['A', 'B'], 'branch_lengths': [0.1, 0.1]},
            'A': {'children': [], 'branch_lengths': []},
            'B': {'children': [], 'branch_lengths': []},
            'C': {'children': [], 'branch_lengths': []}
        }
        self.leaves = ['A', 'B', 'C']
        
    def felsenstein_pruning(self, leaf_data: Dict[str, str], P_matrices: Dict[Tuple[str, float], np.ndarray], 
                           pi: np.ndarray, nucleotides: List[str]) -> Tuple[float, Dict]:
        """
        Felsenstein's pruning algorithm
        Returns: likelihood and partial likelihoods at nodes
        """
        nuc_to_idx = {nuc: i for i, nuc in enumerate(nucleotides)}
        n_states = len(nucleotides)
        
        # Store partial likelihoods
        L = {}
        
        # Post-order traversal (leaves to root)
        def compute_partial_likelihood(node):
            if not self.structure[node]['children']:  # Leaf
                # L[leaf][state] = 1 if observed state matches, else 0
                observed = leaf_data[node]
                L[node] = np.zeros(n_states)
                L[node][nuc_to_idx[observed]] = 1.0
            else:  # Internal node
                L[node] = np.ones(n_states)
                for child, branch_len in zip(self.structure[node]['children'], 
                                             self.structure[node]['branch_lengths']):
                    compute_partial_likelihood(child)
                    P = P_matrices[(child, branch_len)]
                    # L[node][i] *= sum_j P[i,j] * L[child][j]
                    L[node] *= P @ L[child]
            
        compute_partial_likelihood('root')
        
        # Final likelihood: sum over root states weighted by equilibrium frequency
        likelihood = np.sum(pi * L['root'])
        
        return likelihood, L


class PhyloHMM:
    """Phylogenetic Hidden Markov Model for conservation detection"""
    
    def __init__(self, ctmc: ContinuousTimeMarkovChain, tree: PhylogeneticTree):
        self.ctmc = ctmc
        self.tree = tree
        self.states = ['n', 'c']  # neutral, conserved
        self.state_to_idx = {'n': 0, 'c': 1}
        
    def emission_probability(self, column: Dict[str, str], state: str, 
                            Q: np.ndarray, pi: np.ndarray, rho: float = 0.3) -> float:
        """
        P(column | state)
        state 'n': use Q directly
        state 'c': use rho * Q (slower evolution)
        """
        # Build P matrices for each branch
        P_matrices = {}
        
        for node in self.tree.structure:
            for child, branch_len in zip(self.tree.structure[node]['children'],
                                         self.tree.structure[node]['branch_lengths']):
                if state == 'n':
                    Q_use = Q
                else:  # conserved
                    Q_use = rho * Q
                
                P = self.ctmc.transition_probability_matrix(Q_use, branch_len)
                P_matrices[(child, branch_len)] = P
        
        likelihood, _ = self.tree.felsenstein_pruning(column, P_matrices, pi, self.ctmc.nucleotides)
        return likelihood
    
    def forward_algorithm(self, sequence: List[Dict[str, str]], mu: float, nu: float,
                         Q: np.ndarray, pi: np.ndarray, rho: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """Forward with scaling (Siepel et al. 2005)"""
        L = len(sequence)
        alpha = np.zeros((L, 2))
        c = np.zeros(L)
        
        gamma_c = nu / (mu + nu)
        gamma_n = mu / (mu + nu)
        initial_dist = np.array([gamma_n, gamma_c])
        trans = np.array([[1 - mu, mu], [nu, 1 - nu]])
        
        for s_idx, state in enumerate(['n', 'c']):
            emit_prob = self.emission_probability(sequence[0], state, Q, pi, rho)
            alpha[0, s_idx] = initial_dist[s_idx] * emit_prob
        
        c[0] = np.sum(alpha[0, :])
        alpha[0, :] = alpha[0, :] / c[0]
        
        for i in range(1, L):
            for s_idx, state in enumerate(['n', 'c']):
                emit_prob = self.emission_probability(sequence[i], state, Q, pi, rho)
                alpha[i, s_idx] = emit_prob * np.sum(alpha[i-1, :] * trans[:, s_idx])
            
            c[i] = np.sum(alpha[i, :])
            alpha[i, :] = alpha[i, :] / c[i]
        
        log_likelihood = np.sum(np.log(np.maximum(c, 1e-300)))
        return alpha, log_likelihood, c
    
        
    def backward_algorithm(self, sequence: List[Dict[str, str]], mu: float, nu: float,
                          Q: np.ndarray, pi: np.ndarray, rho: float, c: np.ndarray) -> np.ndarray:
        """Backward with correct scaling (Siepel et al. 2005)"""
        L = len(sequence)
        beta = np.zeros((L, 2))
        trans = np.array([[1 - mu, mu], [nu, 1 - nu]])
        
        beta[-1, :] = 1.0
        
        for i in range(L - 2, -1, -1):
            for s_idx, state in enumerate(['n', 'c']):
                for s_next_idx, state_next in enumerate(['n', 'c']):
                    emit_prob = self.emission_probability(sequence[i+1], state_next, Q, pi, rho)
                    beta[i, s_idx] += trans[s_idx, s_next_idx] * emit_prob * beta[i+1, s_next_idx]
            
            beta[i, :] = beta[i, :] / c[i+1]
        
        return beta
    
        
    def posterior_probabilities(self, sequence: List[Dict[str, str]], mu: float, nu: float,
                               Q: np.ndarray, pi: np.ndarray, rho: float) -> np.ndarray:
        """Posterior with scaled Forward-Backward"""
        alpha, _, c = self.forward_algorithm(sequence, mu, nu, Q, pi, rho)
        beta = self.backward_algorithm(sequence, mu, nu, Q, pi, rho, c)
        
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        return gamma
    
        
    def viterbi_algorithm(self, sequence: List[Dict[str, str]], mu: float, nu: float,
                         Q: np.ndarray, pi: np.ndarray, rho: float) -> Tuple[np.ndarray, List[str]]:
        """
        Viterbi algorithm for finding most likely state path
        Returns: delta matrix and path
        """
        L = len(sequence)
        delta = np.zeros((L, 2))
        psi = np.zeros((L, 2), dtype=int)
        
        # Initial distribution
        gamma_c = nu / (mu + nu)
        gamma_n = mu / (mu + nu)
        initial_dist = np.array([gamma_n, gamma_c])
        
        # Transition matrix
        trans = np.array([[1 - mu, mu],
                         [nu, 1 - nu]])
        
        # Initialize
        for s_idx, state in enumerate(['n', 'c']):
            emit_prob = self.emission_probability(sequence[0], state, Q, pi, rho)
            delta[0, s_idx] = np.log(initial_dist[s_idx] + 1e-100) + np.log(emit_prob + 1e-100)
        
        # Recursion
        for i in range(1, L):
            for s_idx, state in enumerate(['n', 'c']):
                emit_prob = self.emission_probability(sequence[i], state, Q, pi, rho)
                scores = delta[i-1, :] + np.log(trans[:, s_idx] + 1e-100)
                psi[i, s_idx] = np.argmax(scores)
                delta[i, s_idx] = np.max(scores) + np.log(emit_prob + 1e-100)
        
        # Backtrack
        path_idx = [np.argmax(delta[-1, :])]
        for i in range(L - 1, 0, -1):
            path_idx.insert(0, psi[i, path_idx[0]])
        
        path = [['n', 'c'][idx] for idx in path_idx]
        
        return delta, path


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sequence_to_columns(sequence: str, leaf_names: List[str]) -> List[Dict[str, str]]:
    """
    IMPORTANT EDUCATIONAL SIMPLIFICATION:
    For pedagogical clarity, this function assumes IDENTICAL sequences across all species.
    In real PhastCons (Siepel et al. 2005), emissions are computed from actual Multiple 
    Sequence Alignments (MSAs) where sequences differ across species.
    
    This simplification allows us to demonstrate the HMM mechanics without requiring 
    complex MSA input.
    """
    # For simplicity, assume sequence is for all species (same sequence)
    columns = []
    for nuc in sequence:
        col = {leaf: nuc for leaf in leaf_names}
        columns.append(col)
    return columns


def plot_matrix_heatmap(matrix: np.ndarray, labels: List[str], title: str):
    """Plot a matrix as heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.4f', xticklabels=labels, 
                yticklabels=labels, cmap='RdYlBu_r', center=0, ax=ax)
    ax.set_title(title)
    return fig


def plot_dataframe_heatmap(df: pd.DataFrame, title: str):
    """Plot DataFrame as heatmap"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title(title)
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("🧬 The Interactive Mathematical PhastCons Lab")
    st.markdown("""
    **An Educational Journey Through Phylogenetic Hidden Markov Models**
    
    This application exposes the complete mathematical engine of the PhastCons algorithm.
    Each module builds on the previous, from basic continuous-time Markov chains to full
    conservation detection via HMMs.
    """)
    
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    modules = [
        "Module 1: Continuous-Time Markov Chain",
        "Module 2: Likelihood of Single Column",
        "Module 3: Two Models (Rate Scaling)",
        "Module 4: HMM Structure",
        "Module 5: Forward-Backward Algorithm",
        "Module 6: EM Algorithm",
        "Module 7: KL Divergence & PIT",
        "Module 8: Viterbi Decoding",
        "Module 9: Complete PhastCons Pipeline"
    ]
    
    selected_module = st.sidebar.radio("Select Module:", modules)
    
    # Initialize core objects
    ctmc = ContinuousTimeMarkovChain()
    tree = PhylogeneticTree()
    
    # ========================================================================
    # MODULE 1: Continuous-Time Markov Chain
    # ========================================================================
    if selected_module == modules[0]:
        st.header("Module 1: Continuous-Time Markov Chain (The Foundation)")
        
        st.markdown("""
        ### Concept
        Evolution is modeled as a **continuous-time Markov chain** with rate matrix $Q$, 
        not just a probability matrix. The probability of transitioning from state $i$ to 
        state $j$ over time $t$ is given by the matrix exponential.
        """)
        
        st.subheader("Mathematical Framework")
        
        st.latex(r"Q = \begin{pmatrix} Q_{AA} & Q_{AC} & Q_{AG} & Q_{AT} \\ Q_{CA} & Q_{CC} & Q_{CG} & Q_{CT} \\ Q_{GA} & Q_{GC} & Q_{GG} & Q_{GT} \\ Q_{TA} & Q_{TC} & Q_{TG} & Q_{TT} \end{pmatrix}")
        
        st.markdown("**Diagonal Constraint:**")
        st.latex(r"Q_{ii} = -\sum_{j \neq i} Q_{ij}")
        
        st.markdown("**Probability Matrix over Time:**")
        st.latex(r"P(t) = e^{Qt} = \sum_{k=0}^{\infty} \frac{(Qt)^k}{k!}")
        
        st.subheader("Interactive Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Select Model:", ["Jukes-Cantor", "HKY85"])
            
            if model_type == "Jukes-Cantor":
                alpha = st.slider("Substitution rate (α):", 0.1, 5.0, 1.0, 0.1)
                Q = ctmc.jukes_cantor_rate_matrix(alpha)
            else:
                kappa = st.slider("Transition/Transversion ratio (κ):", 0.5, 10.0, 2.0, 0.5)
                st.markdown("Equilibrium frequencies (π):")
                pi_A = st.slider("π_A:", 0.1, 0.5, 0.25, 0.05)
                pi_C = st.slider("π_C:", 0.1, 0.5, 0.25, 0.05)
                pi_G = st.slider("π_G:", 0.1, 0.5, 0.25, 0.05)
                pi_T = 1 - pi_A - pi_C - pi_G
                st.write(f"π_T = {pi_T:.2f} (constrained to sum to 1)")
                
                pi = np.array([pi_A, pi_C, pi_G, pi_T])
                Q = ctmc.hky85_rate_matrix(kappa, pi)
        
        with col2:
            t = st.slider("Evolutionary time (t):", 0.01, 2.0, 0.1, 0.01)
        
        st.subheader("Rate Matrix Q")
        Q_df = pd.DataFrame(Q, columns=ctmc.nucleotides, index=ctmc.nucleotides)
        st.dataframe(Q_df.style.format("{:.4f}"))
        
        # Compute P(t)
        P_t = ctmc.transition_probability_matrix(Q, t)
        
        st.subheader(f"Transition Probability Matrix P(t) at t = {t}")
        P_df = pd.DataFrame(P_t, columns=ctmc.nucleotides, index=ctmc.nucleotides)
        st.dataframe(P_df.style.format("{:.4f}"))
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig_Q = plot_matrix_heatmap(Q, ctmc.nucleotides, "Rate Matrix Q")
            st.pyplot(fig_Q)
        
        with col2:
            fig_P = plot_matrix_heatmap(P_t, ctmc.nucleotides, f"P(t={t})")
            st.pyplot(fig_P)
        
        st.info("💡 **Key Insight**: As t increases, P(t) approaches the equilibrium distribution. Try increasing t to see convergence!")
    
    # ========================================================================
    # MODULE 2: Likelihood of Single Column
    # ========================================================================
    elif selected_module == modules[1]:
        st.header("Module 2: Likelihood of a Single Alignment Column")
        
        st.markdown("""
        ### Concept: Felsenstein's Pruning Algorithm
        
        Given a phylogenetic tree and observed nucleotides at the leaves, we calculate
        the likelihood of that observation by marginalizing over all possible ancestral states.
        """)
        
        st.latex(r"P(\mathbf{x} | \psi) = \sum_{\text{ancestral states}} P(\text{leaves, ancestors} | \psi)")
        
        st.subheader("Tree Structure")
        st.code("""
        Tree: ((A:0.1, B:0.1):0.1, C:0.2)
        
            root
           /    \\
          n1     C (branch: 0.2)
         /  \\
        A    B (branches: 0.1 each)
        """)
        
        st.subheader("Interactive Column Likelihood")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nuc_A = st.selectbox("Species A nucleotide:", ctmc.nucleotides, index=0)
            nuc_B = st.selectbox("Species B nucleotide:", ctmc.nucleotides, index=0)
            nuc_C = st.selectbox("Species C nucleotide:", ctmc.nucleotides, index=2)
        
        with col2:
            alpha = st.slider("Substitution rate:", 0.1, 5.0, 1.0, 0.1, key='mod2_alpha')
            model = st.selectbox("Model:", ["Jukes-Cantor"], key='mod2_model')
        
        # Build column data
        column = {'A': nuc_A, 'B': nuc_B, 'C': nuc_C}
        
        # Compute Q and P matrices
        Q = ctmc.jukes_cantor_rate_matrix(alpha)
        pi = ctmc.equilibrium_distribution(Q)
        
        # Build P matrices for each branch
        P_matrices = {}
        for node in tree.structure:
            for child, branch_len in zip(tree.structure[node]['children'],
                                         tree.structure[node]['branch_lengths']):
                P = ctmc.transition_probability_matrix(Q, branch_len)
                P_matrices[(child, branch_len)] = P
        
        # Run Felsenstein's algorithm
        likelihood, partial_L = tree.felsenstein_pruning(column, P_matrices, pi, ctmc.nucleotides)
        
        st.subheader("Results")
        
        st.metric("Column Likelihood", f"{likelihood:.6e}")
        st.metric("Log-Likelihood", f"{np.log(likelihood):.4f}")
        
        st.subheader("Partial Likelihoods at Nodes")
        
        for node in ['A', 'B', 'C', 'n1', 'root']:
            if node in partial_L:
                st.markdown(f"**Node {node}:**")
                L_df = pd.DataFrame({
                    'State': ctmc.nucleotides,
                    'Partial Likelihood': partial_L[node]
                })
                st.dataframe(L_df)
        
        st.info("""
        💡 **Key Insight**: When all leaves have the same nucleotide, the likelihood is higher.
        Try changing one nucleotide to see how the likelihood drops!
        """)
    
    # ========================================================================
    # MODULE 3: Two Models (Rate Scaling)
    # ========================================================================
    elif selected_module == modules[2]:
        st.header("Module 3: Two Models (Conserved vs. Neutral)")
        
        st.markdown(r"""
        ### Concept: Rate Scaling for Conservation
        
        PhastCons uses two models:
        - **Neutral model** $\psi_n$: Standard evolutionary rate (Q)
        - **Conserved model** $\psi_c$: Slower evolution ($\rho Q$ where $\rho < 1$)
        
        The log-likelihood ratio tells us which model better explains the data.
        """)
        
        st.latex(r"\text{Score} = \log \frac{P(\mathbf{x}|\psi_c)}{P(\mathbf{x}|\psi_n)}")
        
        st.subheader("Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rho = st.slider("Conservation scaling (ρ):", 0.01, 1.0, 0.3, 0.01)
            st.markdown(f"- Neutral model: rate = 1.0")
            st.markdown(f"- Conserved model: rate = {rho}")
        
        with col2:
            alpha = st.slider("Base substitution rate:", 0.1, 5.0, 1.0, 0.1, key='mod3_alpha')
        
        st.subheader("Test Column")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            nuc_A = st.selectbox("Species A:", ctmc.nucleotides, index=0, key='mod3_A')
        with col2:
            nuc_B = st.selectbox("Species B:", ctmc.nucleotides, index=0, key='mod3_B')
        with col3:
            nuc_C = st.selectbox("Species C:", ctmc.nucleotides, index=0, key='mod3_C')
        
        column = {'A': nuc_A, 'B': nuc_B, 'C': nuc_C}
        
        # Neutral model
        Q_n = ctmc.jukes_cantor_rate_matrix(alpha)
        pi = ctmc.equilibrium_distribution(Q_n)
        
        P_matrices_n = {}
        for node in tree.structure:
            for child, branch_len in zip(tree.structure[node]['children'],
                                         tree.structure[node]['branch_lengths']):
                P = ctmc.transition_probability_matrix(Q_n, branch_len)
                P_matrices_n[(child, branch_len)] = P
        
        L_neutral, _ = tree.felsenstein_pruning(column, P_matrices_n, pi, ctmc.nucleotides)
        
        # Conserved model
        Q_c = rho * Q_n
        P_matrices_c = {}
        for node in tree.structure:
            for child, branch_len in zip(tree.structure[node]['children'],
                                         tree.structure[node]['branch_lengths']):
                P = ctmc.transition_probability_matrix(Q_c, branch_len)
                P_matrices_c[(child, branch_len)] = P
        
        L_conserved, _ = tree.felsenstein_pruning(column, P_matrices_c, pi, ctmc.nucleotides)
        
        # Results
        st.subheader("Likelihood Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("P(x|ψ_n) Neutral", f"{L_neutral:.6e}")
            st.metric("log P(x|ψ_n)", f"{np.log(L_neutral):.4f}")
        
        with col2:
            st.metric("P(x|ψ_c) Conserved", f"{L_conserved:.6e}")
            st.metric("log P(x|ψ_c)", f"{np.log(L_conserved):.4f}")
        
        with col3:
            log_ratio = np.log(L_conserved) - np.log(L_neutral)
            st.metric("Log-Likelihood Ratio", f"{log_ratio:.4f}")
            
            if log_ratio > 0:
                st.success("✅ Conserved model favored")
            else:
                st.warning("⚠️ Neutral model favored")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        models = ['Neutral', 'Conserved']
        likelihoods = [np.log(L_neutral), np.log(L_conserved)]
        colors = ['blue', 'green']
        ax.bar(models, likelihoods, color=colors, alpha=0.7)
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('Model Comparison')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        st.pyplot(fig)
        
        st.info("""
        💡 **Key Insight**: When all nucleotides are identical (AAA), the conserved model 
        (slower evolution) is favored. Try mixed nucleotides to see the ratio shift!
        """)
    
    # ========================================================================
    # MODULE 4: HMM Structure
    # ========================================================================
    elif selected_module == modules[3]:
        st.header("Module 4: The Hidden Markov Model Structure")
        
        st.markdown("""
        ### Concept: State Transitions
        
        PhastCons uses an HMM with two hidden states:
        - **Neutral (n)**: Evolving at standard rate
        - **Conserved (c)**: Evolving at slower rate
        
        Transitions between states model conserved elements.
        """)
        
        st.subheader("Transition Matrix")
        
        st.latex(r"""
        T = \begin{pmatrix}
        1-\mu & \mu \\
        \nu & 1-\nu
        \end{pmatrix}
        """)
        
        st.markdown(r"""
        where:
        - $\mu$: transition from neutral → conserved
        - $\nu$: transition from conserved → neutral
        """)
        
        st.subheader("Key Statistics")
        
        st.latex(r"\omega = \frac{1}{\mu} \quad \text{(Expected conserved element length)}")
        st.latex(r"\gamma_c = \frac{\nu}{\mu + \nu} \quad \text{(Stationary prob. of conserved state)}")
        
        st.subheader("Interactive Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mu = st.slider("μ (neutral → conserved):", 0.001, 0.5, 0.01, 0.001)
        
        with col2:
            nu = st.slider("ν (conserved → neutral):", 0.001, 0.5, 0.05, 0.001)
        
        # Compute statistics
        omega = 1.0 / mu
        gamma_c = nu / (mu + nu)
        gamma_n = mu / (mu + nu)
        
        # Transition matrix
        T = np.array([[1 - mu, mu],
                      [nu, 1 - nu]])
        
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Element Length (ω)", f"{omega:.2f} bp")
        
        with col2:
            st.metric("Coverage γ_c", f"{gamma_c:.4f}")
        
        with col3:
            st.metric("Coverage γ_n", f"{gamma_n:.4f}")
        
        st.subheader("Transition Matrix T")
        T_df = pd.DataFrame(T, columns=['→ Neutral', '→ Conserved'], 
                           index=['From Neutral', 'From Conserved'])
        st.dataframe(T_df.style.format("{:.4f}"))
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Transition matrix heatmap
        sns.heatmap(T, annot=True, fmt='.4f', xticklabels=['n', 'c'], 
                   yticklabels=['n', 'c'], cmap='Blues', ax=ax1)
        ax1.set_title('Transition Matrix')
        
        # Stationary distribution
        ax2.bar(['Neutral', 'Conserved'], [gamma_n, gamma_c], 
               color=['lightblue', 'lightgreen'], alpha=0.7)
        ax2.set_ylabel('Stationary Probability')
        ax2.set_title('Stationary Distribution')
        ax2.set_ylim([0, 1])
        
        st.pyplot(fig)
        
        st.info("""
        💡 **Key Insight**: 
        - Smaller μ → longer conserved elements
        - Larger ν → shorter conserved elements  
        - Ratio ν/(μ+ν) determines genome coverage
        """)
    
    # ========================================================================
    # MODULE 5: Forward-Backward Algorithm
    # ========================================================================
    elif selected_module == modules[4]:
        st.header("Module 5: Full Likelihood via Forward Algorithm")
        
        st.markdown("""
        ### Concept: Total Probability of a Sequence
        
        The Forward algorithm computes the total probability of observing a sequence
        by summing over all possible hidden state paths.
        """)
        
        st.latex(r"P(\mathbf{x}|\theta) = \sum_{\mathbf{z}} \prod_{i=1}^{L} P(x_i|z_i, \theta) P(z_i|z_{i-1})")
        
        st.markdown("**Forward Variable:**")
        st.latex(r"\alpha_i(s) = P(x_1, \ldots, x_i, z_i = s | \theta)")
        
        st.subheader("Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_input = st.text_input("DNA Sequence:", "AAAGGTTT")
            sequence = sequence_input.upper().strip()
            
            mu = st.slider("μ (transition rate):", 0.001, 0.2, 0.01, 0.001, key='mod5_mu')
            nu = st.slider("ν (transition rate):", 0.001, 0.2, 0.05, 0.001, key='mod5_nu')
        
        with col2:
            rho = st.slider("ρ (conservation):", 0.01, 1.0, 0.3, 0.01, key='mod5_rho')
            alpha_rate = st.slider("Base rate:", 0.1, 3.0, 1.0, 0.1, key='mod5_alpha')
        
        # Validate sequence
        if all(nuc in ctmc.nucleotides for nuc in sequence):
            # Convert sequence to columns (assuming same across species for simplicity)
            columns = sequence_to_columns(sequence, tree.leaves)
            
            # Build models
            Q = ctmc.jukes_cantor_rate_matrix(alpha_rate)
            pi = ctmc.equilibrium_distribution(Q)
            
            # Initialize HMM
            hmm = PhyloHMM(ctmc, tree)
            
            # Run Forward algorithm
            alpha, log_likelihood, c = hmm.forward_algorithm(columns, mu, nu, Q, pi, rho)
            
            st.subheader("Results")
            
            st.metric("Sequence Length", len(sequence))
            st.metric("Total Log-Likelihood", f"{log_likelihood:.4f}")
            
            st.subheader("Forward (α) Table")
            
            alpha_df = pd.DataFrame(
                alpha,
                columns=['Neutral', 'Conserved'],
                index=[f"Pos {i}: {sequence[i]}" for i in range(len(sequence))]
            )
            
            st.dataframe(alpha_df.style.format("{:.6e}"))
            
            # Visualization
            fig = plot_dataframe_heatmap(alpha_df, "Forward Probabilities (α)")
            st.pyplot(fig)
            
            st.info("""
            💡 **Key Insight**: The α values show how probability flows through the HMM.
            Columns where conserved state has higher α are more likely conserved.
            """)
            
        else:
            st.error("Invalid sequence! Use only A, C, G, T.")
    
    # ========================================================================
    # MODULE 6: EM Algorithm
    # ========================================================================
    elif selected_module == modules[5]:
        st.header("Module 6: The EM Algorithm (Parameter Learning)")
        
        st.markdown("""
        ### Concept: Learning Parameters from Data
        
        The Expectation-Maximization (EM) algorithm learns optimal values for μ and ν
        by iteratively:
        1. **E-step**: Compute posterior probabilities P(z|x, θ)
        2. **M-step**: Update parameters to maximize expected likelihood
        """)
        
        st.latex(r"\mu^{new} = \frac{\text{Expected transitions } n \to c}{\text{Expected time in } n}")
        
        st.subheader("Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_input = st.text_input("DNA Sequence:", "AAAAGGGGTTTTT", key='em_seq')
            sequence = sequence_input.upper().strip()
        
        with col2:
            rho = st.slider("ρ (fixed):", 0.01, 1.0, 0.3, 0.01, key='em_rho')
            alpha_rate = st.slider("Base rate (fixed):", 0.1, 3.0, 1.0, 0.1, key='em_alpha')
        
        # Initial parameters
        st.subheader("Initial Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            mu_init = st.slider("Initial μ:", 0.001, 0.1, 0.01, 0.001, key='em_mu_init')
        with col2:
            nu_init = st.slider("Initial ν:", 0.001, 0.1, 0.05, 0.001, key='em_nu_init')
        
        if st.button("Run One EM Iteration"):
            if all(nuc in ctmc.nucleotides for nuc in sequence):
                columns = sequence_to_columns(sequence, tree.leaves)
                Q = ctmc.jukes_cantor_rate_matrix(alpha_rate)
                pi = ctmc.equilibrium_distribution(Q)
                hmm = PhyloHMM(ctmc, tree)
                
                # Initial likelihood
                _, log_lik_before, _ = hmm.forward_algorithm(columns, mu_init, nu_init, Q, pi, rho)
                
                st.subheader("Before EM Step")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("μ", f"{mu_init:.6f}")
                with col2:
                    st.metric("ν", f"{nu_init:.6f}")
                with col3:
                    st.metric("Log-Likelihood", f"{log_lik_before:.4f}")
                
                # E-step: compute posteriors
                gamma = hmm.posterior_probabilities(columns, mu_init, nu_init, Q, pi, rho)
                
                # Compute transition posteriors (xi)
                alpha, _, c_temp = hmm.forward_algorithm(columns, mu_init, nu_init, Q, pi, rho)
                beta = hmm.backward_algorithm(columns, mu_init, nu_init, Q, pi, rho, c_temp)
                
                trans = np.array([[1 - mu_init, mu_init],
                                 [nu_init, 1 - nu_init]])
                
                L = len(columns)
                xi = np.zeros((L - 1, 2, 2))  # [position, from_state, to_state]
                
                for i in range(L - 1):
                    for s1 in range(2):
                        for s2 in range(2):
                            state_name = ['n', 'c'][s2]
                            emit_prob = hmm.emission_probability(columns[i+1], state_name, Q, pi, rho)
                            xi[i, s1, s2] = alpha[i, s1] * trans[s1, s2] * emit_prob * beta[i+1, s2]
                    
                    xi[i] = xi[i] / xi[i].sum()
                
                # M-step: update parameters
                expected_n_to_c = xi[:, 0, 1].sum()
                expected_time_in_n = gamma[:-1, 0].sum()
                
                expected_c_to_n = xi[:, 1, 0].sum()
                expected_time_in_c = gamma[:-1, 1].sum()
                
                mu_new = expected_n_to_c / (expected_time_in_n + 1e-10)
                nu_new = expected_c_to_n / (expected_time_in_c + 1e-10)
                
                # New likelihood
                _, log_lik_after, _ = hmm.forward_algorithm(columns, mu_new, nu_new, Q, pi, rho)
                
                st.subheader("After EM Step")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("μ", f"{mu_new:.6f}", f"{mu_new - mu_init:+.6f}")
                with col2:
                    st.metric("ν", f"{nu_new:.6f}", f"{nu_new - nu_init:+.6f}")
                with col3:
                    st.metric("Log-Likelihood", f"{log_lik_after:.4f}", 
                             f"{log_lik_after - log_lik_before:+.4f}")
                
                # Show posterior probabilities
                st.subheader("Posterior State Probabilities (γ)")
                gamma_df = pd.DataFrame(
                    gamma,
                    columns=['P(Neutral|x)', 'P(Conserved|x)'],
                    index=[f"Pos {i}: {sequence[i]}" for i in range(len(sequence))]
                )
                st.dataframe(gamma_df.style.format("{:.4f}"))
                
                fig = plot_dataframe_heatmap(gamma_df, "Posterior Probabilities")
                st.pyplot(fig)
                
                st.success("✅ EM iteration complete! Likelihood increased." if log_lik_after > log_lik_before 
                          else "⚠️ Likelihood did not increase (possible convergence)")
            else:
                st.error("Invalid sequence!")
    
    # ========================================================================
    # MODULE 7: KL Divergence & PIT
    # ========================================================================
    elif selected_module == modules[6]:
        st.header("Module 7: KL Divergence & Phylogenetic Information Threshold")
        
        st.markdown("""
        ### Concept: Detection Power
        
        The **Phylogenetic Information Threshold (PIT)** determines the minimum alignment 
        length needed to reliably detect conservation. It depends on the KL divergence 
        between conserved and neutral models.
        """)
        
        st.subheader("KL Divergence")
        
        st.latex(r"D_{KL}(\psi_c || \psi_n) = \sum_{\mathbf{x}} P(\mathbf{x}|\psi_c) \log \frac{P(\mathbf{x}|\psi_c)}{P(\mathbf{x}|\psi_n)}")
        
        st.markdown("**Minimum Detectable Length:**")
        st.latex(r"L_{min} \approx \frac{1}{D_{KL}(\psi_c || \psi_n)}")
        
        st.subheader("Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alpha_rate = st.slider("Base rate:", 0.1, 3.0, 1.0, 0.1, key='pit_alpha')
            # Branch lengths are fixed in tree structure
        
        with col2:
            rho_min = st.slider("Min ρ:", 0.01, 0.5, 0.1, 0.01)
            rho_max = st.slider("Max ρ:", 0.5, 1.0, 0.9, 0.01)
        
        # Compute KL divergence for range of rho
        rho_values = np.linspace(rho_min, rho_max, 20)
        kl_divergences = []
        l_mins = []
        
        Q = ctmc.jukes_cantor_rate_matrix(alpha_rate)
        pi = ctmc.equilibrium_distribution(Q)
        
        for rho in rho_values:
            # Compute P(x|conserved) and P(x|neutral) for all possible single columns
            kl = 0.0
            
            # For single-site: consider all 4^3 = 64 possible columns
            for nucs in itertools.product(ctmc.nucleotides, repeat=3):
                column = {'A': nucs[0], 'B': nucs[1], 'C': nucs[2]}
                
                # Neutral
                P_matrices_n = {}
                for node in tree.structure:
                    for child, bl in zip(tree.structure[node]['children'],
                                        tree.structure[node]['branch_lengths']):
                        P = ctmc.transition_probability_matrix(Q, bl)
                        P_matrices_n[(child, bl)] = P
                
                L_n, _ = tree.felsenstein_pruning(column, P_matrices_n, pi, ctmc.nucleotides)
                
                # Conserved
                Q_c = rho * Q
                P_matrices_c = {}
                for node in tree.structure:
                    for child, bl in zip(tree.structure[node]['children'],
                                        tree.structure[node]['branch_lengths']):
                        P = ctmc.transition_probability_matrix(Q_c, bl)
                        P_matrices_c[(child, bl)] = P
                
                L_c, _ = tree.felsenstein_pruning(column, P_matrices_c, pi, ctmc.nucleotides)
                
                # KL contribution
                if L_c > 1e-100 and L_n > 1e-100:
                    kl += L_c * np.log(L_c / L_n)
            
            kl_divergences.append(max(kl, 1e-10))
            l_mins.append(1.0 / max(kl, 1e-10))
        
        st.subheader("Results")
        
        # Plot KL divergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(rho_values, kl_divergences, 'b-', linewidth=2)
        ax1.set_xlabel('Conservation scaling (ρ)')
        ax1.set_ylabel('KL Divergence D(ψ_c || ψ_n)')
        ax1.set_title('KL Divergence vs. Conservation Strength')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(rho_values, l_mins, 'r-', linewidth=2)
        ax2.set_xlabel('Conservation scaling (ρ)')
        ax2.set_ylabel('Minimum Detectable Length (bp)')
        ax2.set_title('Phylogenetic Information Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, min(max(l_mins), 1000)])
        
        st.pyplot(fig)
        
        st.info("""
        💡 **Key Insight**: 
        - As ρ → 1 (weak conservation), KL divergence → 0
        - This means L_min → ∞ (impossible to detect)
        - Stronger conservation (smaller ρ) is easier to detect with shorter alignments
        """)
    
    # ========================================================================
    # MODULE 8: Viterbi Decoding
    # ========================================================================
    elif selected_module == modules[7]:
        st.header("Module 8: Viterbi Decoding (Finding the Most Likely Path)")
        
        st.markdown("""
        ### Concept: Optimal State Path
        
        The Viterbi algorithm finds the most likely sequence of hidden states
        that generated the observed data.
        """)
        
        st.latex(r"\delta_i(k) = \max_{j} \delta_{i-1}(j) \cdot P(k|j) \cdot P(x_i|k)")
        
        st.subheader("Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_input = st.text_input("DNA Sequence:", "AAAAGGGGGTTTT", key='viterbi_seq')
            sequence = sequence_input.upper().strip()
            
            mu = st.slider("μ:", 0.001, 0.2, 0.01, 0.001, key='vit_mu')
            nu = st.slider("ν:", 0.001, 0.2, 0.05, 0.001, key='vit_nu')
        
        with col2:
            rho = st.slider("ρ:", 0.01, 1.0, 0.3, 0.01, key='vit_rho')
            alpha_rate = st.slider("Base rate:", 0.1, 3.0, 1.0, 0.1, key='vit_alpha')
        
        if st.button("Run Viterbi Algorithm"):
            if all(nuc in ctmc.nucleotides for nuc in sequence):
                columns = sequence_to_columns(sequence, tree.leaves)
                Q = ctmc.jukes_cantor_rate_matrix(alpha_rate)
                pi = ctmc.equilibrium_distribution(Q)
                hmm = PhyloHMM(ctmc, tree)
                
                # Run Viterbi
                delta, path = hmm.viterbi_algorithm(columns, mu, nu, Q, pi, rho)
                
                st.subheader("Results")
                
                # Display path
                st.markdown("**Most Likely State Path:**")
                path_str = ''.join([s.upper() for s in path])
                st.code(f"Sequence: {sequence}\nPath:     {path_str}")
                
                # Count conserved elements
                conserved_runs = []
                in_conserved = False
                start = 0
                
                for i, state in enumerate(path):
                    if state == 'c' and not in_conserved:
                        start = i
                        in_conserved = True
                    elif state == 'n' and in_conserved:
                        conserved_runs.append((start, i - 1))
                        in_conserved = False
                
                if in_conserved:
                    conserved_runs.append((start, len(path) - 1))
                
                st.markdown(f"**Number of Conserved Elements:** {len(conserved_runs)}")
                
                if conserved_runs:
                    st.markdown("**Conserved Regions:**")
                    for idx, (s, e) in enumerate(conserved_runs):
                        st.write(f"Element {idx+1}: positions {s}-{e} (length {e-s+1})")
                
                # Viterbi matrix
                st.subheader("Viterbi (δ) Matrix")
                delta_df = pd.DataFrame(
                    delta,
                    columns=['Neutral', 'Conserved'],
                    index=[f"Pos {i}: {sequence[i]}" for i in range(len(sequence))]
                )
                st.dataframe(delta_df.style.format("{:.4f}"))
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Delta heatmap
                sns.heatmap(delta_df, annot=False, cmap='viridis', ax=ax1)
                ax1.set_title('Viterbi Matrix (log-probabilities)')
                
                # Path visualization
                state_numeric = [0 if s == 'n' else 1 for s in path]
                ax2.plot(range(len(sequence)), state_numeric, 'o-', linewidth=2, markersize=8)
                ax2.set_xlabel('Position')
                ax2.set_ylabel('State')
                ax2.set_yticks([0, 1])
                ax2.set_yticklabels(['Neutral', 'Conserved'])
                ax2.set_title('Decoded State Path')
                ax2.grid(True, alpha=0.3)
                
                # Add sequence labels
                for i, nuc in enumerate(sequence):
                    ax2.text(i, state_numeric[i] + 0.1, nuc, ha='center', fontsize=8)
                
                st.pyplot(fig)
                
                st.info("""
                💡 **Key Insight**: The Viterbi path shows the most likely annotation.
                Regions where the conserved state dominates indicate functional elements.
                """)
            else:
                st.error("Invalid sequence!")
    
    # ========================================================================
    # MODULE 9: Complete PhastCons Pipeline
    # ========================================================================
    elif selected_module == modules[8]:
        st.header("Module 9: Complete PhastCons Pipeline")
        
        st.markdown("""
        ### The Full Algorithm: From Sequence to Conservation
        
        This module integrates all 8 previous modules to show how PhastCons works end-to-end:
        
        1. **CTMC Models** (Module 1): Evolution via rate matrix Q
        2. **Column Likelihoods** (Module 2): Felsenstein's pruning for each position
        3. **Two Models** (Module 3): Conserved (ρQ) vs. Neutral (Q) 
        4. **HMM Structure** (Module 4): State transitions (μ, ν)
        5. **Forward Algorithm** (Module 5): Compute total likelihood
        6. **Parameter Estimation** (Module 6): EM algorithm (shown conceptually)
        7. **Detection Threshold** (Module 7): KL divergence and minimum length
        8. **Viterbi Decoding** (Module 8): Most likely conserved elements
        9. **Conservation Scores**: Posterior probabilities at each position
        """)
        
        st.subheader("📊 Complete Pipeline Configuration")
        
        # Input section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Sequence Input**")
            sequence_input = st.text_area(
                "DNA Sequence (A,C,G,T):",
                "AAAAAAGGGGGGTTTTTTAAAACCCCGGGGTTTTAAAA",
                height=100,
                key='pipeline_seq'
            )
            sequence = sequence_input.upper().replace('\n', '').replace(' ', '').strip()
        
        with col2:
            st.markdown("**Evolutionary Parameters**")
            alpha_rate = st.slider("Base substitution rate:", 0.5, 3.0, 1.0, 0.1, key='pipe_alpha')
            rho = st.slider("Conservation scaling (ρ):", 0.05, 0.8, 0.3, 0.05, key='pipe_rho')
            st.info(f"Neutral rate: 1.0\nConserved rate: {rho}")
        
        with col3:
            st.markdown("**HMM Parameters**")
            mu = st.slider("μ (n→c):", 0.01, 0.15, 0.083, 0.001, key='pipe_mu')
            nu = st.slider("ν (c→n):", 0.01, 0.15, 0.030, 0.001, key='pipe_nu')
            omega = 1.0 / mu
            gamma_c = nu / (mu + nu)
            st.info(f"Expected length: {omega:.1f} bp\nExpected coverage: {gamma_c:.3f}")
        
        if st.button("🚀 Run Complete PhastCons Pipeline", key='run_pipeline'):
            if all(nuc in ctmc.nucleotides for nuc in sequence):
                with st.spinner("Running complete PhastCons pipeline..."):
                    
                    # Setup
                    columns = sequence_to_columns(sequence, tree.leaves)
                    Q = ctmc.jukes_cantor_rate_matrix(alpha_rate)
                    pi = ctmc.equilibrium_distribution(Q)
                    hmm = PhyloHMM(ctmc, tree)
                    L = len(sequence)
                    
                    st.success(f"✅ Sequence loaded: {L} positions")
                    
                    # ==========================================
                    # STEP 1: Column Likelihoods
                    # ==========================================
                    st.markdown("---")
                    st.subheader("Step 1: Column-wise Likelihood Computation")
                    st.markdown("Computing P(xᵢ|ψₙ) and P(xᵢ|ψ꜀) for each position using Felsenstein's algorithm")
                    
                    col_likelihoods_n = []
                    col_likelihoods_c = []
                    
                    progress_bar = st.progress(0)
                    for i, col in enumerate(columns):
                        # Neutral
                        P_matrices_n = {}
                        for node in tree.structure:
                            for child, bl in zip(tree.structure[node]['children'],
                                                tree.structure[node]['branch_lengths']):
                                P = ctmc.transition_probability_matrix(Q, bl)
                                P_matrices_n[(child, bl)] = P
                        L_n, _ = tree.felsenstein_pruning(col, P_matrices_n, pi, ctmc.nucleotides)
                        
                        # Conserved
                        Q_c = rho * Q
                        P_matrices_c = {}
                        for node in tree.structure:
                            for child, bl in zip(tree.structure[node]['children'],
                                                tree.structure[node]['branch_lengths']):
                                P = ctmc.transition_probability_matrix(Q_c, bl)
                                P_matrices_c[(child, bl)] = P
                        L_c, _ = tree.felsenstein_pruning(col, P_matrices_c, pi, ctmc.nucleotides)
                        
                        col_likelihoods_n.append(L_n)
                        col_likelihoods_c.append(L_c)
                        
                        progress_bar.progress((i + 1) / L)
                    
                    # Compute log-likelihood ratios
                    log_ratios = [np.log(max(L_c, 1e-100)) - np.log(max(L_n, 1e-100)) 
                                 for L_c, L_n in zip(col_likelihoods_c, col_likelihoods_n)]
                    
                    st.success("✅ Column likelihoods computed")
                    
                    # Show sample
                    sample_df = pd.DataFrame({
                        'Position': range(min(10, L)),
                        'Base': list(sequence[:10]),
                        'P(x|ψₙ)': [f"{x:.6e}" for x in col_likelihoods_n[:10]],
                        'P(x|ψ꜀)': [f"{x:.6e}" for x in col_likelihoods_c[:10]],
                        'Log-Ratio': [f"{x:.4f}" for x in log_ratios[:10]]
                    })
                    st.markdown("**First 10 positions:**")
                    st.dataframe(sample_df)
                    
                    # ==========================================
                    # STEP 2: Forward Algorithm
                    # ==========================================
                    st.markdown("---")
                    st.subheader("Step 2: Forward Algorithm (Computing Total Likelihood)")
                    st.markdown("Computing α table to get P(sequence|θ)")
                    
                    alpha, log_likelihood, c = hmm.forward_algorithm(columns, mu, nu, Q, pi, rho)
                    
                    st.metric("Total Log-Likelihood", f"{log_likelihood:.4f}")
                    
                    # ==========================================
                    # STEP 3: Posterior Probabilities (Conservation Scores)
                    # ==========================================
                    st.markdown("---")
                    st.subheader("Step 3: Posterior Probabilities (Conservation Scores)")
                    st.markdown("Using Forward-Backward to compute P(zᵢ=conserved | sequence)")
                    
                    gamma = hmm.posterior_probabilities(columns, mu, nu, Q, pi, rho)
                    conservation_scores = gamma[:, 1]  # P(conserved | sequence)
                    
                    # ==========================================
                    # STEP 4: Viterbi Decoding
                    # ==========================================
                    st.markdown("---")
                    st.subheader("Step 4: Viterbi Decoding (Most Likely Path)")
                    st.markdown("Finding the optimal state sequence")
                    
                    delta, path = hmm.viterbi_algorithm(columns, mu, nu, Q, pi, rho)
                    
                    # Identify conserved elements
                    conserved_elements = []
                    in_element = False
                    start = 0
                    
                    for i, state in enumerate(path):
                        if state == 'c' and not in_element:
                            start = i
                            in_element = True
                        elif state == 'n' and in_element:
                            # Calculate element score
                            element_score = sum(log_ratios[start:i])
                            conserved_elements.append({
                                'start': start,
                                'end': i - 1,
                                'length': i - start,
                                'sequence': sequence[start:i],
                                'score': element_score,
                                'avg_posterior': np.mean(conservation_scores[start:i])
                            })
                            in_element = False
                    
                    if in_element:
                        element_score = sum(log_ratios[start:])
                        conserved_elements.append({
                            'start': start,
                            'end': L - 1,
                            'length': L - start,
                            'sequence': sequence[start:],
                            'score': element_score,
                            'avg_posterior': np.mean(conservation_scores[start:])
                        })
                    
                    st.success(f"✅ Found {len(conserved_elements)} conserved element(s)")
                    
                    # ==========================================
                    # STEP 5: Detection Threshold (PIT)
                    # ==========================================
                    st.markdown("---")
                    st.subheader("Step 5: Phylogenetic Information Threshold")
                    
                    # Compute TRUE KL divergence over ALL possible columns (like Module 7)
                    kl_div = 0.0
                    
                    for nucs in itertools.product(ctmc.nucleotides, repeat=3):
                        column = {'A': nucs[0], 'B': nucs[1], 'C': nucs[2]}
                        
                        # Neutral model
                        P_matrices_n = {}
                        for node in tree.structure:
                            for child, bl in zip(tree.structure[node]['children'],
                                                tree.structure[node]['branch_lengths']):
                                P = ctmc.transition_probability_matrix(Q, bl)
                                P_matrices_n[(child, bl)] = P
                        L_n, _ = tree.felsenstein_pruning(column, P_matrices_n, pi, ctmc.nucleotides)
                        
                        # Conserved model
                        Q_c = rho * Q
                        P_matrices_c = {}
                        for node in tree.structure:
                            for child, bl in zip(tree.structure[node]['children'],
                                                tree.structure[node]['branch_lengths']):
                                P = ctmc.transition_probability_matrix(Q_c, bl)
                                P_matrices_c[(child, bl)] = P
                        L_c, _ = tree.felsenstein_pruning(column, P_matrices_c, pi, ctmc.nucleotides)
                        
                        if L_c > 1e-300 and L_n > 1e-300:
                            kl_div += L_c * np.log2(L_c / L_n)
                    
                    # Compute Lmin
                    if mu < 1 and nu < 1 and kl_div > 0:
                        # Paper-consistent approximation (Siepel et al. 2005)
                        L_min = 1.0 / kl_div
                    else:
                        L_min = float('inf')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("KL Divergence", f"{kl_div:.4f} bits/site")
                    with col2:
                        st.metric("Min Length (Lₘᵢₙ)", f"{L_min:.1f} bp" if L_min < 1000 else "∞")
                    with col3:
                        pit = L_min * kl_div if L_min < 1000 else float('inf')
                        st.metric("PIT", f"{pit:.2f} bits" if pit < 1000 else "∞")
                    
                    # ==========================================
                    # RESULTS SUMMARY
                    # ==========================================
                    st.markdown("---")
                    st.subheader("📋 Results Summary")
                    
                    # Conservation elements table
                    if conserved_elements:
                        st.markdown("**Predicted Conserved Elements:**")
                        elements_df = pd.DataFrame([
                            {
                                'Element': f"CE{i+1}",
                                'Start': el['start'],
                                'End': el['end'],
                                'Length': el['length'],
                                'Sequence': el['sequence'][:20] + '...' if el['length'] > 20 else el['sequence'],
                                'Score': f"{el['score']:.2f}",
                                'Avg Posterior': f"{el['avg_posterior']:.4f}"
                            }
                            for i, el in enumerate(conserved_elements)
                        ])
                        st.dataframe(elements_df)
                    else:
                        st.warning("No conserved elements predicted with current parameters")
                    
                    # ==========================================
                    # VISUALIZATIONS
                    # ==========================================
                    st.markdown("---")
                    st.subheader("📊 Comprehensive Visualizations")
                    
                    # Create comprehensive plot
                    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
                    positions = np.arange(L)
                    
                    # Plot 1: Log-likelihood ratios
                    axes[0].plot(positions, log_ratios, 'b-', linewidth=1.5, label='Log[P(x|ψ꜀)/P(x|ψₙ)]')
                    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
                    axes[0].set_ylabel('Log-Likelihood Ratio')
                    axes[0].set_title('Column-wise Evidence for Conservation')
                    axes[0].legend(loc='upper right')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot 2: Conservation scores (posterior probabilities)
                    axes[1].fill_between(positions, 0, conservation_scores, 
                                        color='green', alpha=0.5, label='P(conserved|data)')
                    axes[1].plot(positions, conservation_scores, 'g-', linewidth=1.5)
                    axes[1].axhline(y=gamma_c, color='red', linestyle='--', 
                                   label=f'Prior γ={gamma_c:.3f}', alpha=0.7)
                    axes[1].set_ylabel('Conservation Score')
                    axes[1].set_ylim([0, 1])
                    axes[1].set_title('Posterior Conservation Probabilities (Forward-Backward)')
                    axes[1].legend(loc='upper right')
                    axes[1].grid(True, alpha=0.3)
                    
                    # Plot 3: Viterbi path
                    viterbi_numeric = [1 if s == 'c' else 0 for s in path]
                    axes[2].fill_between(positions, 0, viterbi_numeric, 
                                        color='orange', alpha=0.6, step='mid')
                    axes[2].set_ylabel('State')
                    axes[2].set_yticks([0, 1])
                    axes[2].set_yticklabels(['Neutral', 'Conserved'])
                    axes[2].set_title('Viterbi Path (Most Likely State Sequence)')
                    axes[2].grid(True, alpha=0.3)
                    
                    # Highlight conserved elements
                    for el in conserved_elements:
                        axes[2].axvspan(el['start'], el['end'], color='red', alpha=0.2)
                    
                    # Plot 4: Sequence with annotations
                    axes[3].text(0.5, 0.7, sequence, fontsize=8, family='monospace',
                               ha='center', va='center', transform=axes[3].transAxes)
                    path_str = ''.join([s.upper() for s in path])
                    axes[3].text(0.5, 0.3, path_str, fontsize=8, family='monospace',
                               ha='center', va='center', transform=axes[3].transAxes,
                               color='red')
                    axes[3].set_xlim([0, L])
                    axes[3].set_ylim([0, 1])
                    axes[3].axis('off')
                    axes[3].set_title('Sequence and Annotation (Red = Conserved)')
                    
                    axes[3].set_xlabel('Position in Sequence')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # ==========================================
                    # INTERPRETATION
                    # ==========================================
                    st.markdown("---")
                    st.subheader("🔍 Interpretation Guide")
                    
                    st.markdown("""
                    **How to Read These Results:**
                    
                    1. **Log-Likelihood Ratios (Top)**: Positive values favor conservation at that position
                       - Higher peaks = stronger conservation signal
                       - Negative values favor neutral evolution
                    
                    2. **Conservation Scores (Second)**: Posterior probability P(conserved | data)
                       - Values near 1 = high confidence in conservation
                       - Dashed line shows prior expectation (γ)
                       - Deviation from prior shows data influence
                    
                    3. **Viterbi Path (Third)**: Most likely discrete annotation
                       - Orange = Conserved state
                       - White = Neutral state
                       - Red boxes = Predicted conserved elements
                    
                    4. **Sequence Annotation (Bottom)**: Direct visual alignment
                       - Top row = Original sequence
                       - Bottom row = State annotation (C=Conserved, N=Neutral)
                    
                    **Key Insights:**
                    - PhastCons integrates phylogenetic evidence across positions
                    - The HMM smooths noisy column-wise signals
                    - Conservation scores balance prior and likelihood
                    - Viterbi provides discrete element boundaries
                    """)
                    
                    # ==========================================
                    # ALGORITHM FLOW DIAGRAM
                    # ==========================================
                    st.markdown("---")
                    st.subheader("🔄 Complete Algorithm Flow")
                    
                    st.code("""
PHASTCONS ALGORITHM FLOW
========================

INPUT: Multiple alignment, phylogenetic tree, parameters (Q, ρ, μ, ν)

STEP 1: For each alignment column i:
    ├─ Compute P(xᵢ|ψₙ) using Felsenstein's algorithm with Q
    └─ Compute P(xᵢ|ψ꜀) using Felsenstein's algorithm with ρQ

STEP 2: Run Forward Algorithm:
    ├─ Initialize: α₀(c) = γ꜀·P(x₀|ψ꜀), α₀(n) = γₙ·P(x₀|ψₙ)
    ├─ Recursion: αᵢ(s) = P(xᵢ|s) · Σⱼ αᵢ₋₁(j)·P(s|j)
    └─ Log-Likelihood: log P(X|θ) = Σᵢ log(cᵢ)

STEP 3: Run Backward Algorithm:
    ├─ Initialize: βₗ(c) = βₗ(n) = 1
    └─ Recursion: βᵢ(s) = Σₛ' P(s'|s)·P(xᵢ₊₁|s')·βᵢ₊₁(s')

STEP 4: Compute Posteriors:
    └─ γᵢ(s) = P(zᵢ=s|X) = αᵢ(s)·βᵢ(s) / P(X|θ)
        ↳ Conservation Score at position i

STEP 5: Run Viterbi Algorithm:
    ├─ Forward: δᵢ(s) = max_j [δᵢ₋₁(j)·P(s|j)]·P(xᵢ|s)
    ├─ Backtrack: Find path maximizing δₗ(s)
    └─ Output: Most likely state sequence

STEP 6: Identify Conserved Elements:
    ├─ Find runs of consecutive 'conserved' states
    ├─ Compute element scores: Σᵢ log[P(xᵢ|ψ꜀)/P(xᵢ|ψₙ)]
    └─ Filter by minimum length (Lₘᵢₙ)

STEP 7 (Optional): EM Parameter Refinement:
    ├─ E-step: Compute expected counts using γᵢ(s)
    └─ M-step: Update Q, ρ, (μ, ν if not constrained)

OUTPUT: 
    ├─ Conservation scores (posterior probabilities)
    ├─ Predicted conserved elements (positions, scores)
    └─ Annotation (conserved vs. neutral at each site)
                    """, language='text')
                    
                    st.success("""
                    ✅ **Pipeline Complete!** 
                    
                    You've now seen how PhastCons integrates:
                    - Phylogenetic models (CTMC)
                    - Hidden Markov Models (state transitions)
                    - Dynamic programming (Forward-Backward, Viterbi)
                    - Statistical inference (posterior probabilities)
                    
                    Try different parameters to see how they affect predictions!
                    """)
                    
            else:
                st.error("Invalid sequence! Use only A, C, G, T.")
        
        st.info("""
        💡 **Module 9 Summary**: This is the complete PhastCons algorithm in action.
        Every step builds on the previous modules:
        - Modules 1-3 provide the phylogenetic foundation
        - Module 4 adds the HMM structure
        - Modules 5-6 enable inference and learning
        - Module 7 establishes detection thresholds
        - Module 8 produces final annotations
        
        Together, they form a powerful framework for detecting functional conservation
        in genomic sequences!
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    Created as a part Computational Functional Genomics course January 2026 at *IISER Pune*.
    
    If you detect an error, email jatin.raghuwanshi@students.iiserpune.ac.in 


    **Reference**: Siepel et al. (2005) Genome Research
    """)


if __name__ == "__main__":
    main()
