# OT_ASSIGN
# Optimization Techniques Assignment — OT_ASSIGN

This repository contains two Python implementations of classic Operations Research optimization algorithms, each solving a well-known constrained optimization case study, as required by the assignment brief:

1. **`ot_big_m_simplex.py`** — Linear Programming via the **Big-M Simplex Method**
2. **`tsp.py`** — Transportation Problem via **Vogel's Approximation Method (VAM) + MODI Method**

Both programs take the problem data (objective/cost coefficients, constraints, supply/demand) as **user input** at runtime rather than hardcoding a single fixed problem, and print the optimal solution.

---

## 1. `ot_big_m_simplex.py` — Big-M Simplex Method

### Problem type

A profit-maximization Linear Programming Problem (LPP) with **two decision variables** and **three constraints** — two resource limits (`≤`) and one minimum-requirement constraint (`≥`):

```
Maximize   Z = c1*x1 + c2*x2
Subject to:
    a11*x1 + a12*x2 <= b1
    a21*x1 + a22*x2 <= b2
    a31*x1 + a32*x2 >= b3
    x1, x2 >= 0
```

The `≥` constraint is what makes this a genuine **Big-M** problem: it can't be satisfied by the origin (0, 0), so a plain slack-only starting basis isn't feasible, and an artificial variable is needed to get the simplex method started.

### Converting to standard form

| Constraint          | Extra variable(s) added                              |
|---------------------|--------------------------------------------------------|
| `≤` (constraint 1)  | slack `s1` (coefficient +1)                            |
| `≤` (constraint 2)  | slack `s2` (coefficient +1)                            |
| `≥` (constraint 3)  | surplus `s3` (coefficient −1) **and** artificial `A1` (coefficient +1) |

The six standard-form variables are `x1, x2, s1, s2, s3, A1`. The artificial variable `A1` is given a huge penalty cost, `M = 1,000,000`, with a **negative** sign in the (maximization) objective:

```
Max Z = c1*x1 + c2*x2 + 0*s1 + 0*s2 + 0*s3 - M*A1
```

so the simplex algorithm is driven to push `A1` out of the solution (down to 0) as quickly as possible — if it can't (i.e. `A1 > 0` at the end), the original problem is infeasible.

### How the code works

- **User input:** prompts for the two objective coefficients, then each constraint's two coefficients and its right-hand side.
- **`make_tableau(A, b, c, basis)`:** builds the initial simplex tableau `[A | b]` plus an objective row, and makes that row "canonical" (zero coefficients under the starting basic variables `s1, s2, A1`) by adding back each basic variable's cost contribution.
- **`simplex(T, basis)`:** the standard simplex loop —
  - *Entering variable* = the most negative coefficient in the objective row.
  - *Leaving variable* = the minimum-ratio test (`RHS / pivot column`, skipping non-positive entries) to preserve feasibility.
  - *Pivot* = normalize the pivot row and eliminate that column from every other row.
  - Repeats until no negative coefficient remains (optimal), or raises an "Unbounded solution" error if a column has no valid ratio.
- **Solution extraction:** reads each basic variable's value from the final RHS column, checks that the artificial variable `A1` is 0 (feasibility check), then reports `x1`, `x2`, and the maximum profit (computed from the entered `c1`, `c2`, so it stays correct for whatever coefficients are typed in).

### Example run

Input:
```
40 30
2 1 100
1 2 80
1 1 20
```
(i.e. `Max Z = 40x1 + 30x2` s.t. `2x1+x2≤100`, `x1+2x2≤80`, `x1+x2≥20`)

Output:
```
Optimal Solution:
x1 = 40.0
x2 = 20.0
Maximum Profit = 2200.0
```

This result was independently verified against `scipy.optimize.linprog` (HiGHS solver), which returns the same optimum.

### Usage

```bash
python3 ot_big_m_simplex.py
```
Then enter the objective coefficients and each constraint's data when prompted.

---

## 2. `tsp.py` — Transportation Problem (VAM + MODI)

### Problem type

A classic distribution problem: a set of **sources** with fixed supply capacities must ship goods to a set of **destinations** with fixed demands, at a given per-unit cost, minimizing total transportation cost.

```
Minimize   Z = sum( cost[i][j] * x[i][j] )
Subject to:
    sum_j x[i][j] = supply[i]     for every source i
    sum_i x[i][j] = demand[j]     for every destination j
    x[i][j] >= 0
```

### How the code works

- **`balance(cost, supply, demand)`:** if total supply ≠ total demand, adds a zero-cost dummy source or destination so the problem is balanced (a requirement of the transportation algorithm).
- **`VAM(cost, supply, demand)` — Stage 1, initial basic feasible solution:**
  - For each active row/column, computes a **penalty** (difference between the two lowest costs in it).
  - Picks the row/column with the **highest penalty**, allocates as much as possible to its **cheapest cell**, updates the remaining supply/demand, and "crosses out" any row/column that's exhausted.
  - Repeats until all supply and demand are allocated. VAM typically produces a starting solution close to optimal.
- **`find_cycle(alloc, start, basic_set)`:** given a non-basic cell, finds the unique closed loop through the current basic cells by alternating row-moves and column-moves — the "stepping-stone" path needed to reallocate.
- **`MODI(cost, alloc)` — Stage 2, optimality test and improvement:**
  - Treats every positive-allocation cell as **basic** (padding with zero-allocation cells when the solution is degenerate, so there are always exactly `m + n − 1` basic cells and the dual system is solvable).
  - Computes dual values `u[i], v[j]` from `u[i] + v[j] = cost[i][j]` on every basic cell.
  - Computes the **opportunity cost** `Δ = cost[i][j] − (u[i] + v[j])` for every non-basic cell. If every Δ ≥ 0, the allocation is optimal.
  - Otherwise, enters the cell with the most negative Δ, shifts the maximum possible quantity around its closed loop (alternately adding/subtracting), and swaps the entering cell in for whichever cell hits zero allocation.
  - Repeats until the optimality test passes.

### Example run

Input:
```
3
4
19 30 50 10
70 30 40 60
40 8 70 20
7 9 18
5 8 7 14
```
(3 sources, 4 destinations, the classic textbook cost/supply/demand matrix)

Output:
```
Optimal Allocation Matrix:
 [[ 5  0  0  2]
 [ 0  2  7  0]
 [ 0  6  0 12]]
Minimum Cost: 743
```

Shipment plan: Source 1 → Dest 1 (5 units), Source 1 → Dest 4 (2), Source 2 → Dest 2 (2), Source 2 → Dest 3 (7), Source 3 → Dest 2 (6), Source 3 → Dest 4 (12). This result was cross-checked against an independently written reference implementation of VAM + MODI.

### Usage

```bash
python3 tsp.py
```
Then enter the number of sources, number of destinations, the cost matrix (row by row), the supply values, and the demand values when prompted.

---

## Requirements

```
numpy
```
```bash
pip install numpy
```

## Repository structure

| File                    | Method                          | Problem solved                          |
|-------------------------|----------------------------------|------------------------------------------|
| `ot_big_m_simplex.py`   | Big-M Simplex Method              | Constrained LPP (max profit, `≤`/`≥` mix) |
| `tsp.py`                | Vogel's Approximation Method + MODI | Transportation / distribution problem   |

## Summary

Both scripts implement a two-step computational workflow rather than a hand-solved answer:

- **`ot_big_m_simplex.py`** converts a constrained LPP into standard form (adding slack, surplus, and artificial variables as needed), then solves it via Big-M simplex tableau iterations to find the optimal `x1`, `x2`, and objective value.
- **`tsp.py`** finds an initial basic feasible solution via VAM, then tests and improves it for optimality via the MODI method, arriving at the minimum-cost shipment plan.

Both were validated against independent solvers (`scipy.optimize.linprog` for the LPP, and a separate reference VAM/MODI implementation for the transportation problem) to confirm correctness of the reported optimal values.
