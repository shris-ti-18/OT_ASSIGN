import numpy as np

cost = np.array([
    [19, 30, 50, 10],
    [70, 30, 40, 60],
    [40,  8, 70, 20]
], dtype=float)

supply = [7, 9, 18]
demand = [5, 8, 7, 14]


def balance(cost, supply, demand):
    """Add a zero-cost dummy row/column if total supply != total demand."""
    cost = cost.copy()
    supply = list(supply)
    demand = list(demand)
    ts, td = sum(supply), sum(demand)
    if ts == td:
        return cost, supply, demand
    if ts < td:
        supply.append(td - ts)
        cost = np.vstack([cost, np.zeros((1, cost.shape[1]))])
    else:
        demand.append(ts - td)
        cost = np.hstack([cost, np.zeros((cost.shape[0], 1))])
    return cost, supply, demand


def VAM(cost, supply, demand):
    m, n = cost.shape
    s, d = supply.copy(), demand.copy()
    alloc = np.zeros((m, n))
    active_r, active_c = [True] * m, [True] * n

    while any(active_r) and any(active_c):
        row_pen = [-1] * m
        col_pen = [-1] * n

        for i in range(m):
            if active_r[i]:
                vals = sorted([cost[i][j] for j in range(n) if active_c[j]])
                row_pen[i] = vals[1] - vals[0] if len(vals) >= 2 else (vals[0] if len(vals) == 1 else -1)

        for j in range(n):
            if active_c[j]:
                vals = sorted([cost[i][j] for i in range(m) if active_r[i]])
                col_pen[j] = vals[1] - vals[0] if len(vals) >= 2 else (vals[0] if len(vals) == 1 else -1)

        max_row, max_col = max(row_pen), max(col_pen)

        if max_row >= max_col:
            i = row_pen.index(max_row)
            j = min([j for j in range(n) if active_c[j]], key=lambda x: cost[i][x])
        else:
            j = col_pen.index(max_col)
            i = min([i for i in range(m) if active_r[i]], key=lambda x: cost[x][j])

        q = min(s[i], d[j])
        alloc[i][j] = q
        s[i] -= q
        d[j] -= q

        if s[i] == 0 and d[j] == 0:
            active_r[i] = False
            if sum(active_r) > 0:
                pass
            else:
                active_c[j] = False
        elif s[i] == 0:
            active_r[i] = False
        else:
            active_c[j] = False

    return alloc


def find_cycle(alloc, start, basic_set):
    m, n = alloc.shape

    def search(path, move_row):
        curr = path[-1]
        i, j = curr

        if move_row:
            candidates = [(i, jj) for jj in range(n) if jj != j]
        else:
            candidates = [(ii, j) for ii in range(m) if ii != i]

        for cell in candidates:
            if cell == start and len(path) >= 4:
                return path
            if cell in basic_set and cell not in path:
                res = search(path + [cell], not move_row)
                if res:
                    return res
        return None

    return search([start], True) or search([start], False)


def MODI(cost, alloc):
    m, n = cost.shape

    basic = set(zip(*np.where(alloc > 0)))
    required = m + n - 1
    if len(basic) < required:
        for i in range(m):
            for j in range(n):
                if len(basic) >= required:
                    break
                if (i, j) not in basic:
                    basic.add((i, j))
            if len(basic) >= required:
                break
    # -----------------------------------------------------------------

    while True:
        u, v = [None] * m, [None] * n
        u[0] = 0

        changed = True
        while changed:
            changed = False
            for i, j in basic:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i][j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i][j] - v[j]
                    changed = True

      
        u = [x if x is not None else 0 for x in u]
        v = [x if x is not None else 0 for x in v]

        delta = np.full((m, n), np.nan)
        for i in range(m):
            for j in range(n):
                if (i, j) not in basic:
                    delta[i][j] = cost[i][j] - u[i] - v[j]

        min_delta = np.nanmin(delta)
        if min_delta >= -1e-9:
            break

        enter = np.unravel_index(np.nanargmin(delta), delta.shape)
        path = find_cycle(alloc, enter, basic)

        minus_cells = path[1::2]
        theta = min(alloc[i][j] for i, j in minus_cells)

        for k, (i, j) in enumerate(path):
            if k % 2 == 0:
                alloc[i][j] += theta
            else:
                alloc[i][j] -= theta

        basic.add(enter)
        leaving = next(cell for cell in minus_cells if alloc[cell] == 0)
        basic.remove(leaving)

    return alloc


# Run program
cost_b, supply_b, demand_b = balance(cost, supply, demand)
allocation = VAM(cost_b, supply_b, demand_b)
optimal_allocation = MODI(cost_b, allocation.copy())
min_cost = np.sum(cost_b * optimal_allocation)

print("Optimal Allocation Matrix:\n", optimal_allocation.astype(int))
print("Minimum Cost:", int(min_cost))