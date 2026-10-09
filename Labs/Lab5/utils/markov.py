import numpy as np

def markov(rho, A, nmax, rng):
    """
    Simulate a discrete-time homogeneous Markov chain.
    """
    # 1. Verifications
    A = np.asarray(A)
    rho = np.asarray(rho)
    
    N = len(rho)
    assert A.shape == (N, N), "Transition matrix A must be square and match rho size"
    assert np.allclose(np.sum(A, axis=1), 1), "Rows of A must sum to 1 (stochastic matrix)"
    assert np.all(A >= 0), "Transition probabilities in A must be non-negative"
    assert np.isclose(np.sum(rho), 1), "Initial distribution rho must sum to 1"
    assert np.all(rho >= 0), "Initial probabilities in rho must be non-negative"
    
    # 2. Simulation
    states = np.arange(N)
    X = np.zeros(nmax + 1, dtype=int)
    
    # Initial state
    X[0] = rng.choice(states, p=rho)
    
    # Transitions
    for q in range(nmax):
        X[q+1] = rng.choice(states, p=A[X[q]])
        
    return X

# Wrapper function to pass multiple arguments to markov
def simulate_chain_parallel(seed, rho, A, nmax):
    rng_local = np.random.default_rng(seed)
    return markov(rho, A, nmax, rng_local)