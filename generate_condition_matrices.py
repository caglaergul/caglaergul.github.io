import numpy as np
import pandas as pd
from itertools import permutations

# Define action names
actions = ['No Usage', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
collections = ['Low Collection', 'Medium Collection', 'High Collection']

def create_monotonic_row(start_val, end_val, length, fixed_positions=None):
    """Create a monotonic increasing row with optional fixed positions."""
    row = np.zeros(length)

    if fixed_positions:
        for pos, val in fixed_positions.items():
            row[pos] = val

    # Fill unfixed positions
    unfixed = [i for i in range(length) if i not in (fixed_positions or {})]

    if not unfixed:
        return row

    # Determine ranges for each unfixed position
    for i, pos in enumerate(unfixed):
        # Lower bound
        if pos == 0:
            lower = start_val
        else:
            lower = row[pos - 1] + 0.01

        # Upper bound
        if pos == length - 1:
            upper = end_val
        else:
            # Check if next position is fixed
            next_fixed = None
            for j in range(pos + 1, length):
                if j in (fixed_positions or {}) or row[j] > 0:
                    next_fixed = (j, row[j])
                    break

            if next_fixed:
                steps = next_fixed[0] - pos
                upper = next_fixed[1] - 0.01 * steps
            else:
                upper = end_val

        if lower >= upper:
            upper = lower + 0.01

        row[pos] = np.random.uniform(lower, min(upper, end_val))

    return row

def generate_constrained_matrix(min_val, max_val, fixed_cols=None, fixed_cells=None):
    """
    Generate a 3x6 matrix with monotonic constraints.
    fixed_cols: dict of {col_idx: [val1, val2, val3]}
    fixed_cells: dict of {(row, col): val}
    """
    matrix = np.zeros((3, 6))

    # Apply fixed columns
    if fixed_cols:
        for col_idx, vals in fixed_cols.items():
            matrix[:, col_idx] = vals

    # Apply fixed cells
    if fixed_cells:
        for (r, c), val in fixed_cells.items():
            matrix[r, c] = val

    # Generate each row
    for row in range(3):
        fixed_in_row = {}
        for col in range(6):
            if fixed_cols and col in fixed_cols:
                fixed_in_row[col] = matrix[row, col]
            elif fixed_cells and (row, col) in fixed_cells:
                fixed_in_row[col] = matrix[row, col]

        # Determine start and end values for row
        start = fixed_in_row.get(0, min_val)
        end = fixed_in_row.get(5, max_val)

        matrix[row, :] = create_monotonic_row(start, end, 6, fixed_in_row)

    # Ensure column-wise monotonicity
    for col in range(6):
        for row in range(1, 3):
            if matrix[row, col] <= matrix[row - 1, col]:
                matrix[row, col] = matrix[row - 1, col] + np.random.uniform(0.01, 10)

    # Cap values
    matrix = np.clip(matrix, min_val, max_val)

    # Reapply fixed values after capping
    if fixed_cols:
        for col_idx, vals in fixed_cols.items():
            matrix[:, col_idx] = vals
    if fixed_cells:
        for (r, c), val in fixed_cells.items():
            matrix[r, c] = val

    return matrix

def generate_smart_matrices():
    """Generate matrices with intelligent parameter selection."""
    # Benefit: No Usage = 0, varying values
    benefit = generate_constrained_matrix(
        min_val=100,
        max_val=1000,
        fixed_cols={0: [0, 0, 0]}
    )

    # Cost: No Usage = x (same for all rows)
    x = np.random.uniform(150, 350)
    cost = generate_constrained_matrix(
        min_val=x,
        max_val=1000,
        fixed_cols={0: [x, x, x]}
    )

    # Breach: No Usage = 0.75, High Collection - Very High = 0.99
    breach = generate_constrained_matrix(
        min_val=0.75,
        max_val=0.99,
        fixed_cols={0: [0.75, 0.75, 0.75]},
        fixed_cells={(2, 5): 0.99}
    )

    return benefit, cost, breach

def check_conditions(benefit, cost, breach):
    """Check if matrices satisfy all conditions."""
    # Verify basic constraints
    if not np.all(benefit[:, 0] == 0):
        return False, None, None, None, None

    if not np.all(cost[:, 0] == cost[0, 0]):
        return False, None, None, None, None

    if not np.all(breach[:, 0] == 0.75) or breach[2, 5] != 0.99:
        return False, None, None, None, None

    # Check monotonicity
    for matrix in [benefit, cost, breach]:
        # Row-wise
        for i in range(3):
            if not np.all(np.diff(matrix[i, :]) >= -1e-6):
                return False, None, None, None, None
        # Column-wise
        for j in range(6):
            if not np.all(np.diff(matrix[:, j]) >= -1e-6):
                return False, None, None, None, None

    # Calculate payoffs
    expected_payoff = benefit - breach * cost
    worst_case_payoff = benefit - cost

    # Find maximizers (excluding No Usage)
    ep_max = np.argmax(expected_payoff[:, 1:], axis=1) + 1
    wc_max = np.argmax(worst_case_payoff[:, 1:], axis=1) + 1

    # Check: WC maximizers are {3, 4, 5}
    if set(wc_max) != {3, 4, 5}:
        return False, None, None, None, None

    # Check: EP maximizers are 2 apart from WC maximizers
    for i in range(3):
        if abs(ep_max[i] - wc_max[i]) != 2:
            return False, None, None, None, None

    return True, expected_payoff, worst_case_payoff, ep_max, wc_max

def print_matrix(matrix, title):
    """Print matrix in formatted table."""
    df = pd.DataFrame(matrix, index=collections, columns=actions)
    print(f"\n{title}:")
    print(df.to_string(float_format=lambda x: f'{x:.2f}'))

def main():
    """Generate 10 valid solutions."""
    print("Generating matrices that satisfy all conditions...")
    print("This may take a minute...\n")

    valid_solutions = []
    attempts = 0
    max_attempts = 2000000  # Increase limit

    while len(valid_solutions) < 10 and attempts < max_attempts:
        attempts += 1

        benefit, cost, breach = generate_smart_matrices()
        valid, ep, wc, ep_max, wc_max = check_conditions(benefit, cost, breach)

        if valid:
            valid_solutions.append({
                'benefit': benefit,
                'cost': cost,
                'breach': breach,
                'expected_payoff': ep,
                'worst_case_payoff': wc,
                'ep_maximizers': ep_max,
                'wc_maximizers': wc_max
            })
            print(f"✓ Found solution {len(valid_solutions)}/10 (after {attempts} attempts)")

        if attempts % 100000 == 0:
            print(f"  ... still searching ({attempts} attempts, {len(valid_solutions)} found)")

    print()
    if len(valid_solutions) < 10:
        print(f"⚠ Warning: Only found {len(valid_solutions)} solutions after {max_attempts} attempts\n")
    else:
        print(f"✓ Successfully found all 10 solutions!\n")

    # Print all solutions
    for idx, sol in enumerate(valid_solutions, 1):
        print("=" * 80)
        print(f"SOLUTION {idx}")
        print("=" * 80)

        print_matrix(sol['benefit'], "BENEFIT MATRIX")
        print_matrix(sol['cost'], "COST MATRIX")
        print_matrix(sol['breach'], "BREACH PROBABILITY MATRIX")
        print_matrix(sol['expected_payoff'], "EXPECTED PAYOFF MATRIX")
        print_matrix(sol['worst_case_payoff'], "WORST CASE PAYOFF MATRIX")

        print("\n" + "-" * 80)
        print("ROW-WISE MAXIMIZERS:")
        print("-" * 80)
        for i, coll in enumerate(collections):
            print(f"{coll}:")
            print(f"  Worst Case Payoff:  {actions[sol['wc_maximizers'][i]]}")
            print(f"  Expected Payoff:    {actions[sol['ep_maximizers'][i]]}")
        print()

if __name__ == "__main__":
    main()
