import os
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import glob

def read_hypergraph(path):
    """Read a hypergraph from a .dtl file and return list of edges."""
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            left = line.find("(")
            right = line.find(")")
            if left == -1 or right == -1:
                continue

            inside = line[left + 1:right]
            inside = inside.replace(",", " ")
            parts = inside.split()
            vertices = [int(p) for p in parts]
            edges.append(vertices)
    return edges


def solve_min_edge_cover(edges):
    """Solve minimum edge cover via ILP. Return (chosen_edges, optimum_value)."""
    n_edges = len(edges)
    vertices = sorted({v for e in edges for v in e})

    prob = LpProblem("min_edge_cover", LpMinimize)

    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n_edges)]
    prob += lpSum(x[i] for i in range(n_edges))

    for v in vertices:
        incident = [i for i, e in enumerate(edges) if v in e]
        prob += lpSum(x[i] for i in incident) >= 1

    prob.solve(PULP_CBC_CMD(msg=False))

    chosen = [i for i in range(n_edges) if x[i].value() is not None and x[i].value() > 0.5]
    optimum = prob.objective.value()
    return chosen, optimum


if __name__ == "__main__":
    # collect all .dtl files in current directory
    dtl_files = sorted(glob.glob("*.dtl"))

    for path in dtl_files:
        edges = read_hypergraph(path)
        chosen, optimum = solve_min_edge_cover(edges)

        print(f"File: {path}")
        print("  Number of edges:", len(edges))
        print("  Optimal edge cover size:", optimum)
        print("  Chosen edge indices:", chosen)
        print()