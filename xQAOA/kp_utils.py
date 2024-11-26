import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


def generate_strongly_correlated(n=10):
    weights = np.random.randint(1, 1001, n)
    values = weights + 1000
    return values, weights


def generate_inversely_strongly_correlated(n=10):
    values = np.random.randint(1, 1001, n)
    weights = values + np.random.choice([98, 102], n)
    return values, weights


def generate_profit(n=10):
    weights = np.random.randint(1, 1001, n)
    values = 3 * np.ceil(weights / 3).astype(int)
    return values, weights


def generate_strong_spanner(n=10):
    span_size = 20
    span_values, span_weights = generate_strongly_correlated(span_size)
    span_weights = np.ceil(2 * span_weights / 3).astype(int)
    span_values = np.ceil(2 * span_values / 3).astype(int)
    
    values, weights = [], []
    for _ in range(n):
        idx = np.random.randint(0, span_size)
        s = np.random.choice([1, 2, 3])
        values.append(s * span_values[idx])
        weights.append(s * span_weights[idx])
    
    return np.array(values), np.array(weights)


def generate_profit_spanner(n=10):
    span_size = 20
    span_values, span_weights = generate_profit(span_size)
    span_weights = np.ceil(2 * span_weights / 3).astype(int)
    span_values = np.ceil(2 * span_values / 3).astype(int)
    
    values, weights = [], []
    for _ in range(n):
        idx = np.random.randint(0, span_size)
        s = np.random.choice([1, 2, 3])
        values.append(s * span_values[idx])
        weights.append(s * span_weights[idx])
    
    return np.array(values), np.array(weights)


def lazy_greedy_knapsack(v, w, c):
    """
    Very Greedy algorithm for the Knapsack Problem.
    Items are selected purely based on the highest efficiency ratio.
    """
    r = np.array(v) / np.array(w)  # Efficiency ratio

    # Sort items by efficiency ratio in descending order
    indices = np.argsort(-r)
    
    total_value = 0
    total_weight = 0
    selected_items = []

    for i in indices:
        if total_weight + w[i] <= c:
            total_weight += w[i]
            total_value += v[i]
            selected_items.append(i)

    # Create the bitstring representing the selected items
    bitstring = np.zeros(len(v), dtype=int)
    bitstring[selected_items] = 1

    return total_value, total_weight, ''.join(map(str, bitstring))


def lazy_greedy_knapsack(v, w, c):
    """
    Lazy Greedy algorithm for the Knapsack Problem.
    Implements Algorithm 1 from the paper exactly.
    """
    n = len(v)
    # Calculate efficiency ratios
    r = np.array(v) / np.array(w)
    
    # Sort indices by efficiency ratio (descending)
    # In case of ties, smaller index gets priority (stable sort)
    indices = np.argsort(-r, kind='stable')
    
    # Initialize variables as per the paper
    c_prime = c - w[indices[0]]  # Key difference: subtract first item's weight immediately
    j = 1  # Start from second item since we considered first item
    x = np.zeros(n, dtype=int)
    
    # Main loop following paper's algorithm
    while c_prime > 0 and j < n:
        x[indices[j]] = 1
        j += 1
        if j < n:
            c_prime = c_prime - w[indices[j]]
            
    # Calculate final value and weight
    value = sum(x[i] * v[i] for i in range(n))
    weight = sum(x[i] * w[i] for i in range(n))
    
    return value, weight, ''.join(map(str, x))

def very_greedy_knapsack(v, w, c):
    """
    Very Greedy algorithm for the Knapsack Problem.
    Implements Algorithm 2 from the paper exactly.
    """
    n = len(v)
    # Calculate efficiency ratios
    r = np.array(v) / np.array(w)
    
    # Sort indices by efficiency ratio (descending)
    # In case of ties, smaller index gets priority (stable sort)
    indices = np.argsort(-r, kind='stable')
    
    # Initialize variables as per the paper
    c_prime = c
    j = 0
    x = np.zeros(n, dtype=int)
    
    # Main loop following paper's algorithm
    while c_prime > 0 and j < n:
        # Inner while loop to skip items that don't fit
        while j < n and c_prime < w[indices[j]]:
            j += 1
            
        if j < n:  # If we found an item that fits
            x[indices[j]] = 1
            c_prime = c_prime - w[indices[j]]
            j += 1
    
    # Calculate final value and weight
    value = sum(x[i] * v[i] for i in range(n))
    weight = sum(x[i] * w[i] for i in range(n))
    
    return value, weight, ''.join(map(str, x))


def plot_rank_and_ratio(results, methods=None, labels=None):
    # Extract distributions
    distributions = list(results['very_greedy'].keys())
    distributions_cleaned = [d.replace('generate_', '') for d in distributions]

    # Initialize data for plotting
    if methods == None:
        methods = ['lazy_greedy', 'very_greedy', 'hourglass', 'copula', 'X']
        labels = ['LG', 'VG', r'$QKP_{H}$', r'$QKP_{COP}$', r'$QKP_{X}$']
    num_methods = len(methods)
    bar_width = 0.15  # Adjusted width for better spacing

    rank_data = {method: [results[method][dist]['rank_solution'] for dist in distributions] for method in methods}
    ratio_data = {method: [results[method][dist]['ratio_optim'] for dist in distributions] for method in methods}

    x = np.arange(len(distributions_cleaned))  # X-axis positions for distributions

    # Updated professional color palette
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', 'k']  # Blue, Green, Red, Purple

    ### Plot 1: Rank of Each Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    for i, (method, label) in enumerate(zip(methods, labels)):
        bars = ax1.bar(
            x + i * bar_width,
            rank_data[method],
            width=bar_width,
            label=label,
            alpha=0.85,
            color=colors[i],
            edgecolor='black'
        )
        
        # Add text labels above each bar for rank
        for bar, rank in zip(bars, rank_data[method]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(rank),
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

    ax1.set_title('Rank of Each Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distribution', fontsize=12)
    ax1.set_ylabel('Rank', fontsize=12)
    ax1.set_xticks(x + bar_width * (num_methods - 1) / 2)
    ax1.set_xticklabels(distributions_cleaned, rotation=0, ha='center')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()

    # Show the first plot
    plt.show()

    ### Plot 2: Ratio to Optimal for Each Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    for i, (method, label) in enumerate(zip(methods, labels)):
        bars = ax2.bar(
            x + i * bar_width,
            ratio_data[method],
            width=bar_width,
            label=label,
            alpha=0.85,
            color=colors[i],
            edgecolor='black'
        )
        
        # Add text labels above each bar for ratio
        for bar, ratio in zip(bars, ratio_data[method]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{ratio:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

    ax2.set_title('Ratio to Optimal for Each Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distribution', fontsize=12)
    ax2.set_ylabel('Ratio to Optimal', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(x + bar_width * (num_methods - 1) / 2)
    ax2.set_xticklabels(distributions_cleaned, rotation=0, ha='center')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout()

    # Show the second plot
    plt.show()


def bruteforce_knapsack(values, weights, capacity, bit_mapping="regular", show_progress=True):
    """
    Brute-force solver for the knapsack problem.

    Parameters:
    values (list): List of item values.
    weights (list): List of item weights.
    capacity (int): Maximum weight capacity of the knapsack.
    bit_mapping (str): Either "regular" or "inverse" for bit interpretation.
    show_progress (bool): Whether to show the progress bar.

    Returns:
    list: Ranked solutions as a list of tuples (value, weight, bitstring).
    """
    import itertools
    from tqdm import tqdm

    n = len(values)
    ranked_solutions = []
    total_combinations = 2 ** n

    # Select iteration tool based on progress bar option
    iterator = (
        tqdm(itertools.product([0, 1], repeat=n), 
             total=total_combinations, 
             desc="Evaluating knapsack combinations")
        if show_progress else itertools.product([0, 1], repeat=n)
    )

    for subset in iterator:
        if bit_mapping == "regular":
            total_weight = sum(weights[i] * subset[i] for i in range(n))
            total_value = sum(values[i] * subset[i] for i in range(n))
        elif bit_mapping == "inverse":
            total_weight = sum(weights[i] * (1 - subset[i]) for i in range(n))
            total_value = sum(values[i] * (1 - subset[i]) for i in range(n))

        if total_weight <= capacity:
            ranked_solutions.append((total_value, total_weight, subset))
        else:
            ranked_solutions.append((0, 0, subset))

    ranked_solutions.sort(key=lambda x: x[0], reverse=True)
    ranked_solutions = [
        (int(value), int(weight), ''.join(map(str, bitstring))) 
        for value, weight, bitstring in ranked_solutions
    ]

    return ranked_solutions

def compute_min_D(B, C, L):
    """
    Compute the smallest D such that the capacity is non-negative.
    
    Parameters:
    B: list or np.array of B coefficients
    C: list or np.array of C coefficients
    L: threshold value for capacity
    
    Returns:
    optimal_D: The minimum D that satisfies the condition
    """
    B = np.array(B)
    C = np.array(C)
    
    # Compute the numerator and denominator
    numerator = 2 * L + np.sum(B / C)
    denominator = np.sum(1 / C)
    
    # Calculate D
    optimal_D = numerator / denominator
    return optimal_D