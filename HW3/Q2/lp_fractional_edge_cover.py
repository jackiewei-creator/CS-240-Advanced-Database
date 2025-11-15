import os
from pulp import LpProblem, LpVariable, LpMinimize, PULP_CBC_CMD, value

def parse_dtl(path):
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("c") or line.startswith("#"):
                continue
            if "(" in line and ")" in line:
                inside = line[line.index("(") + 1 : line.index(")")]
                if inside.strip() == "":
                    edges.append([])
                else:
                    verts = [int(x) for x in inside.split(",")]
                    edges.append(verts)
    return edges

def fractional_edge_cover(edges):
    if not edges:
        return None, None

    vertices = sorted({v for e in edges for v in e})

    m = len(edges)
    prob = LpProblem("fractional_edge_cover", LpMinimize)

    x = [LpVariable("x_%d" % i, lowBound=0) for i in range(m)]

    prob += sum(x)

    for v in vertices:
        prob += sum(x[i] for i, e in enumerate(edges) if v in e) >= 1

    solver = PULP_CBC_CMD(msg=False, timeLimit=60)
    prob.solve(solver)

    if prob.status != 1:
        return None, None

    obj = value(prob.objective)
    sol = [value(var) for var in x]
    return obj, sol

def main():
    folder = "."
    files = sorted(f for f in os.listdir(folder) if f.endswith(".dtl"))

    for fname in files:
        path = os.path.join(folder, fname)
        edges = parse_dtl(path)
        obj, sol = fractional_edge_cover(edges)

        print("File:", fname)
        print("  Number of edges:", len(edges))

        if obj is None:
            print("  Solver failed or timed out.")
            print()
            continue

        print("  Optimal fractional edge cover value: %.4f" % obj)

        chosen = [(i, w) for i, w in enumerate(sol) if w > 1e-6]
        print("  Edges with positive weight:")
        for idx, w in chosen:
            print("    Edge %d: weight %.4f, vertices %s" % (idx, w, edges[idx]))
        print()

if __name__ == "__main__":
    main()