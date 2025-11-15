from pulp import LpProblem, LpVariable, LpMinimize, LpStatus

edges = {
    0: {0, 1},
    1: {1, 2},
    2: {0, 2}
}

bags = [0, 1]   # try 2 bags

model = LpProblem("triangle_ghd_width1", LpMinimize)

# x[e][b] = 1 if edge e goes to bag b
x = {e:{b:LpVariable(f"x_{e}_{b}",0,1,cat="Binary") for b in bags} for e in edges}

# each edge must be in exactly one bag
for e in edges:
    model += sum(x[e][b] for b in bags) == 1

# Correct width ≤ 1 constraint:
# each bag may contain at most ONE hyperedge
for b in bags:
    model += sum(x[e][b] for e in edges) <= 1

model += 0  # dummy objective

result = model.solve()

print("ILP status:", LpStatus[result])
if LpStatus[result] == "Optimal":
    print("\nValid assignment found (unexpected):")
    for e in edges:
        for b in bags:
            if x[e][b].value() > 0.5:
                print(f"  Edge {e} assigned to bag {b}")
else:
    print("\nNo solution→ Triangle is a counterexample for width = 1.")