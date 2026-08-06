import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. NETWORK SEIR SIMULATOR
# ==========================================

def calculate_beta(R0, gamma, graph):
    """
    Calculates edge transmission probability (beta) from R0 for a network 
    using the degree heterogeneity ratio: R0 approx (beta / gamma) * (<k^2> - <k>) / <k>
    """
    degrees = np.array([d for _, d in graph.degree()])
    k_mean = np.mean(degrees)
    k2_mean = np.mean(degrees ** 2)
    
    if k2_mean - k_mean <= 0:
        return 0.0
    
    beta = (R0 * gamma * k_mean) / (k2_mean - k_mean)
    return min(beta, 1.0) # Probability cap

def simulate_seir_network(graph, R0, incubation_period, infectious_period, num_days, initial_infected=5):
    """
    Simulates stochastic SEIR epidemic spread on a graph.
    Returns: daily incidence array (newly exposed per day).
    """
    sigma = 1.0 / incubation_period  # Progression rate E -> I
    gamma = 1.0 / infectious_period  # Recovery rate I -> R
    
    # Calculate transmission probability per edge
    beta = calculate_beta(R0, gamma, graph)
    
    # States: 0 = Susceptible, 1 = Exposed, 2 = Infected, 3 = Recovered
    states = {node: 0 for node in graph.nodes()}
    
    # Seed initial infections
    seed_nodes = np.random.choice(list(graph.nodes()), size=initial_infected, replace=False)
    for node in seed_nodes:
        states[node] = 2
        
    incidence = np.zeros(num_days)
    
    for day in range(num_days):
        new_states = states.copy()
        daily_new_exposed = 0
        
        # Get currently infected and exposed nodes
        infected_nodes = [n for n, s in states.items() if s == 2]
        exposed_nodes = [n for n, s in states.items() if s == 1]
        
        # 1. S -> E (Infection step)
        for inf_node in infected_nodes:
            for neighbor in graph.neighbors(inf_node):
                if states[neighbor] == 0 and new_states[neighbor] == 0:
                    if np.random.rand() < beta:
                        new_states[neighbor] = 1
                        daily_new_exposed += 1
                        
        # 2. E -> I (Incubation step)
        for exp_node in exposed_nodes:
            if np.random.rand() < sigma:
                new_states[exp_node] = 2
                
        # 3. I -> R (Recovery step)
        for inf_node in infected_nodes:
            if np.random.rand() < gamma:
                new_states[inf_node] = 3
                
        states = new_states
        incidence[day] = daily_new_exposed
        
    return incidence

# ==========================================
# 2. ABC FITTING PROCEDURE
# ==========================================

def distance_metric(sim_data, empirical_data):
    """Euclidean distance between simulated and empirical incidence curve."""
    return np.sqrt(np.sum((sim_data - empirical_data) ** 2))

def abc_rejection(empirical_incidence, graph, incubation_period, infectious_period, 
                  prior_range=(1.0, 5.0), num_samples=1000, tolerance=50.0):
    """
    ABC Rejection Sampler to estimate R0 posterior distribution.
    """
    num_days = len(empirical_incidence)
    accepted_R0 = []
    distances = []
    
    print(f"Starting ABC Sampling ({num_samples} draws)...")
    
    for i in range(num_samples):
        # Sample R0 candidate from Uniform Prior
        R0_cand = np.random.uniform(prior_range[0], prior_range[1])
        
        # Run simulation
        sim_incidence = simulate_seir_network(
            graph, R0_cand, incubation_period, infectious_period, num_days
        )
        
        # Calculate distance
        dist = distance_metric(sim_incidence, empirical_incidence)
        
        # Reject / Accept
        if dist <= tolerance:
            accepted_R0.append(R0_cand)
            distances.append(dist)
            
        if (i + 1) % (num_samples // 5) == 0:
            print(f"  Progress: {i+1}/{num_samples} iterations | Accepted: {len(accepted_R0)}")
            
    return np.array(accepted_R0), np.array(distances)

# ==========================================
# 3. RUN EXAMPLE WITH SYNTHETIC DATA
# ==========================================

if __name__ == "__main__":
    np.random.seed(42)

    # --- User Parameters ---
    N = 1000                  # Population (Nodes)
    m = 3                     # Scale-free graph parameter (Barabási-Albert)
    incubation_period = 4.0   # Days
    infectious_period = 5.0   # Days
    true_R0 = 2.5             # True R0 for synthetic data generation
    num_days = 60
    
    # 1. Build Scale-Free Network
    network = nx.barabasi_albert_graph(N, m, seed=42)
    
    # 2. Generate Synthetic Empirical Incidence Data
    print("Generating synthetic empirical incidence...")
    empirical_incidence = simulate_seir_network(
        network, true_R0, incubation_period, infectious_period, num_days
    )
    
    # 3. Run ABC Fitting
    accepted_posterior, distances = abc_rejection(
        empirical_incidence=empirical_incidence,
        graph=network,
        incubation_period=incubation_period,
        infectious_period=infectious_period,
        prior_range=(1.0, 5.0),
        num_samples=2000,
        tolerance=35.0  # Adjust depending on noise tolerance
    )

    # 4. Results & Summary
    print("\n--- ABC Estimation Summary ---")
    print(f"True R0: {true_R0}")
    if len(accepted_posterior) > 0:
        print(f"Estimated R0 Mean: {np.mean(accepted_posterior):.3f}")
        print(f"95% CI: [{np.percentile(accepted_posterior, 2.5):.3f}, {np.percentile(accepted_posterior, 97.5):.3f}]")
    else:
        print("No samples accepted. Consider increasing 'tolerance' or 'num_samples'.")