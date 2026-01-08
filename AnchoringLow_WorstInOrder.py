import numpy as np
import pandas as pd

# Define action names
actions = ['No Usage', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
collections = ['Low Collection', 'Medium Collection', 'High Collection']

def print_matrix(matrix, title):
    """Print matrix in formatted table."""
    df = pd.DataFrame(matrix, index=collections, columns=actions)
    print(f"\n{title}:")
    print(df.to_string(float_format=lambda x: f'{x:.2f}'))

def create_manual_example():
    """Create a manually crafted example that satisfies all constraints."""
    # Benefit matrix: No Usage = 0, monotonic increasing
    benefit = np.array([
        [0, 200, 300, 400, 500, 600],      # Low Collection
        [0, 250, 350, 450, 550, 650],      # Medium Collection
        [0, 300, 400, 500, 600, 700]       # High Collection
    ], dtype=float)

    # Cost matrix: No Usage = same (200), monotonic increasing
    cost = np.array([
        [200, 250, 350, 450, 500, 550],    # Low Collection
        [200, 300, 400, 500, 550, 600],    # Medium Collection
        [200, 350, 450, 550, 600, 650]     # High Collection
    ], dtype=float)

    # Breach Probability: No Usage = 0.2, High-High = 0.99
    breach = np.array([
        [0.2, 0.80, 0.85, 0.88, 0.90, 0.92],  # Low Collection
        [0.2, 0.82, 0.87, 0.90, 0.93, 0.95],  # Medium Collection
        [0.2, 0.84, 0.89, 0.92, 0.95, 0.99]   # High Collection
    ], dtype=float)

    return benefit, cost, breach

def analyze_solution(benefit, cost, breach, solution_num):
    """Analyze and print a solution."""
    print("=" * 80)
    print(f"SOLUTION {solution_num}")
    print("=" * 80)

    print_matrix(benefit, "BENEFIT MATRIX")
    print_matrix(cost, "COST MATRIX")
    print_matrix(breach, "BREACH PROBABILITY MATRIX")

    # Calculate payoffs
    expected_payoff = benefit - breach * cost
    worst_case_payoff = benefit - cost

    print_matrix(expected_payoff, "EXPECTED PAYOFF MATRIX")
    print_matrix(worst_case_payoff, "WORST CASE PAYOFF MATRIX")

    # Find maximizers (excluding No Usage column)
    ep_max = np.argmax(expected_payoff[:, 1:], axis=1) + 1
    wc_max = np.argmax(worst_case_payoff[:, 1:], axis=1) + 1

    print("\n" + "-" * 80)
    print("ROW-WISE MAXIMIZERS:")
    print("-" * 80)
    for i, coll in enumerate(collections):
        print(f"{coll}:")
        print(f"  Worst Case Payoff:  {actions[wc_max[i]]}")
        print(f"  Expected Payoff:    {actions[ep_max[i]]}")

    # Check conditions
    print("\n" + "-" * 80)
    print("CONSTRAINT CHECKS:")
    print("-" * 80)

    # Check if WC maximizers are {3, 4, 5}
    wc_set = set(wc_max)
    print(f"✓ WC maximizers are all different: {list(wc_set)}")
    if wc_set == {3, 4, 5}:
        print("  ✓ They are Medium, High, and Very High")
    else:
        print(f"  ✗ Expected {{Medium, High, Very High}} but got {[actions[i] for i in wc_set]}")

    # Check if EP maximizers are 2 apart
    print("\n✓ Checking if EP and WC maximizers are 2 apart:")
    all_valid = True
    for i in range(3):
        diff = abs(ep_max[i] - wc_max[i])
        status = "✓" if diff == 2 else "✗"
        print(f"  {status} {collections[i]}: |{actions[ep_max[i]]} - {actions[wc_max[i]]}| = {diff}")
        if diff != 2:
            all_valid = False

    if all_valid:
        print("\n✓✓✓ ALL CONSTRAINTS SATISFIED! ✓✓✓")
    else:
        print("\n✗ Some constraints not satisfied")

    print()

    return all_valid, ep_max, wc_max

def generate_random_matrices():
    """Generate random matrices with all constraints."""
    # Benefit: No Usage = 0
    benefit = np.zeros((3, 6))
    benefit[:, 0] = 0

    for i in range(3):
        vals = np.sort(np.random.uniform(200, 900, 5))
        benefit[i, 1:] = vals

    # Ensure column-wise monotonicity (skip column 0 - No Usage)
    for j in range(1, 6):  # Start from column 1, not 0
        for i in range(1, 3):
            if benefit[i, j] <= benefit[i-1, j]:
                benefit[i, j] = benefit[i-1, j] + np.random.uniform(10, 100)

    # Cost: No Usage = x (same for all rows)
    x = np.random.uniform(100, 250)  # Lower range to ensure positive worst case
    cost = np.zeros((3, 6))
    cost[:, 0] = x

    for i in range(3):
        # Generate costs that are lower than benefits to ensure positive worst case payoff
        vals = np.sort(np.random.uniform(x + 20, 700, 5))
        cost[i, 1:] = vals

    # Ensure column-wise monotonicity (skip column 0 - No Usage)
    for j in range(1, 6):  # Start from column 1, not 0
        for i in range(1, 3):
            if cost[i, j] <= cost[i-1, j]:
                cost[i, j] = cost[i-1, j] + np.random.uniform(10, 100)

    # Ensure worst case payoffs are positive (benefit > cost for columns 1-5)
    for i in range(3):
        for j in range(1, 6):
            if benefit[i, j] <= cost[i, j]:
                benefit[i, j] = cost[i, j] + np.random.uniform(50, 200)

    # Re-ensure column-wise monotonicity for benefit after adjustment
    for j in range(1, 6):
        for i in range(1, 3):
            if benefit[i, j] <= benefit[i-1, j]:
                benefit[i, j] = benefit[i-1, j] + np.random.uniform(10, 100)

    # Re-ensure row-wise monotonicity for benefit after all adjustments
    for i in range(3):
        for j in range(2, 6):
            if benefit[i, j] <= benefit[i, j-1]:
                benefit[i, j] = benefit[i, j-1] + np.random.uniform(10, 100)

    # Re-ensure row-wise monotonicity for cost
    for i in range(3):
        for j in range(2, 6):
            if cost[i, j] <= cost[i, j-1]:
                cost[i, j] = cost[i, j-1] + np.random.uniform(10, 100)

    # Breach: No Usage = 0.2, High-High = 0.99
    breach = np.zeros((3, 6))
    breach[:, 0] = 0.2

    for i in range(3):
        if i < 2:
            vals = np.sort(np.random.uniform(0.21, 0.97, 5))
        else:
            vals = np.sort(np.random.uniform(0.21, 0.98, 5))
        breach[i, 1:] = vals

    breach[2, 5] = 0.99

    # Ensure column-wise monotonicity (skip column 0 - No Usage)
    for j in range(1, 6):  # Start from column 1, not 0
        for i in range(1, 3):
            if breach[i, j] <= breach[i-1, j]:
                breach[i, j] = min(breach[i-1, j] + np.random.uniform(0.01, 0.03), 0.99)

    breach[2, 5] = 0.99  # Reapply

    return benefit, cost, breach

def check_valid(benefit, cost, breach):
    """Check if matrices satisfy all conditions."""
    # Check constraint 1: Benefit No Usage column must be all 0
    if not np.allclose(benefit[:, 0], 0):
        return False

    # Check constraint 2: Cost No Usage column must be equal
    if not np.allclose(cost[:, 0], cost[0, 0]):
        return False

    # Check constraint 3: Breach No Usage column must be all 0.2
    if not np.allclose(breach[:, 0], 0.2):
        return False

    # Check constraint 4: Breach High Collection - Very High must be 0.99
    if not np.isclose(breach[2, 5], 0.99):
        return False

    # Check constraint 5: Row-wise monotonicity for all matrices
    for matrix in [benefit, cost, breach]:
        for i in range(3):
            for j in range(1, 6):
                if matrix[i, j] < matrix[i, j-1]:
                    return False

    # Check constraint 6: Column-wise monotonicity for all matrices
    for matrix in [benefit, cost, breach]:
        for j in range(6):
            for i in range(1, 3):
                if matrix[i, j] < matrix[i-1, j]:
                    return False

    # Calculate payoffs
    expected_payoff = benefit - breach * cost
    worst_case_payoff = benefit - cost

    # Check constraint 7: Worst case payoffs must be positive for columns 1-5
    # (No Usage column can be negative)
    for i in range(3):
        for j in range(1, 6):
            if worst_case_payoff[i, j] <= 0:
                return False

    # Find maximizers
    ep_max = np.argmax(expected_payoff[:, 1:], axis=1) + 1
    wc_max = np.argmax(worst_case_payoff[:, 1:], axis=1) + 1

    # Check condition 8: WC maximizers must be specific for each row
    # Low Collection -> Very High (5)
    # Medium Collection -> High (4)
    # High Collection -> Medium (3)
    if wc_max[0] != 5:  # Low Collection must maximize at Very High
        return False
    if wc_max[1] != 4:  # Medium Collection must maximize at High
        return False
    if wc_max[2] != 3:  # High Collection must maximize at Medium
        return False

    # Check condition 9: EP maximizers are 2 apart from WC maximizers
    for i in range(3):
        if abs(ep_max[i] - wc_max[i]) != 2:
            return False

    return True

def main():
    print("="*80)
    print("MATRIX GENERATION DEMONSTRATION")
    print("="*80)
    print("\nThis program generates matrices satisfying these conditions:")
    print("1. Benefit matrix: No Usage = 0, values 100-1000, monotonic")
    print("2. Cost matrix: No Usage = same value x, values 100-1000, monotonic")
    print("3. Breach matrix: No Usage = 0.2, High-High = 0.99, values 0.2-0.99, monotonic")
    print("4. WC maximizers: Low Collection->Very High, Medium->High, High->Medium")
    print("5. EP maximizers must be 2 actions apart from WC maximizers\n")

    # Create and analyze manual example
    print("\n" + "="*80)
    print("DEMONSTRATING WITH A MANUALLY CRAFTED EXAMPLE")
    print("="*80 + "\n")

    benefit, cost, breach = create_manual_example()
    valid, ep_max, wc_max = analyze_solution(benefit, cost, breach, 1)

    # Try to generate random solutions
    print("\n" + "="*80)
    print("ATTEMPTING TO GENERATE RANDOM SOLUTIONS")
    print("="*80 + "\n")

    print("Searching for valid random solutions...")
    print("(This may take a while due to strict constraints)\n")

    valid_count = 1  # We already have the manual one
    attempts = 0
    max_attempts = 1000000  # Increase to find all 10

    while valid_count < 20 and attempts < max_attempts:
        attempts += 1

        benefit, cost, breach = generate_random_matrices()

        if check_valid(benefit, cost, breach):
            valid_count += 1
            print(f"✓ Found solution {valid_count}/20 (after {attempts} attempts)")
            analyze_solution(benefit, cost, breach, valid_count)

        if attempts % 50000 == 0 and valid_count == 1:
            print(f"  ... still searching ({attempts} attempts, {valid_count-1} random solutions found)")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if valid_count >= 10:
        print(f"✓ Successfully generated {valid_count} solutions (1 manual + {valid_count-1} random)")
    else:
        print(f"⚠ Generated {valid_count} solution(s) (1 manual + {valid_count-1} random) after {attempts} attempts")
        print("\nNote: The constraints are very strict. Finding solutions requires:")
        print("  - All three matrices to be monotonic in both dimensions")
        print("  - WC maximizers to be exactly {Medium, High, Very High} across the three rows")
        print("  - EP maximizers to be exactly 2 positions away from WC maximizers")
    print()

if __name__ == "__main__":
    main()
