import numpy as np

M = 1000000

# Objective: Max Z = 40x1 + 30x2 - M*A1
# Variables: x1, x2, s1, s2, s3, A1

c = np.array([40, 30, 0, 0, 0, -M], dtype=float)

# Constraint matrix
A = np.array([
    [2, 1, 1, 0, 0, 0],
    [1, 2, 0, 1, 0, 0],
    [1, 1, 0, 0, -1, 1]
], dtype=float)

b = np.array([100, 80, 20], dtype=float)

# Initial basic variables
basis = [2, 3, 5]       # s1, s2, A1

def make_tableau(A, b, c, basis):
    m, n = A.shape
    T = np.zeros((m + 1, n + 1))

    T[:m, :n] = A
    T[:m, n] = b

    # Objective row
    T[m, :n] = -c

    # Make objective row canonical
    for i in range(m):
        T[m] += c[basis[i]] * T[i]

    return T

def simplex(T, basis):
    m = len(basis)
    n = T.shape[1] - 1

    while True:
        # Entering variable: most negative coefficient
        col = np.argmin(T[m, :n])

        if T[m, col] >= -1e-9:
            break

        # Ratio test
        ratios = []

        for i in range(m):
            if T[i, col] > 1e-9:
                ratios.append(T[i, n] / T[i, col])
            else:
                ratios.append(np.inf)

        row = np.argmin(ratios)

        if ratios[row] == np.inf:
            raise Exception("Unbounded solution")

        # Pivot
        pivot = T[row, col]
        T[row] /= pivot

        for i in range(m + 1):
            if i != row:
                T[i] -= T[i, col] * T[row]

        basis[row] = col

    return T, basis


# Create initial tableau
T = make_tableau(A, b, c, basis)

# Apply Big-M simplex
T, basis = simplex(T, basis)

# Extract solution
solution = np.zeros(len(c))

for i, var in enumerate(basis):
    solution[var] = T[i, -1]

# Check artificial variable
if solution[5] > 1e-6:
    print("No feasible solution exists.")
else:
    print("Optimal Solution:")
    print("x1 =", solution[0])
    print("x2 =", solution[1])
    print("Maximum Profit =", 40 * solution[0] + 30 * solution[1])