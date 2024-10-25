import gurobipy as gp
import numpy as np
import torch 
from gurobipy import GRB
from math import log
from .knapsack_utils import get_ordering, normalize
from functools import lru_cache
import pdb

@lru_cache(maxsize=1000)
def ip_enumerate(counts: tuple, elements: tuple, num_solutions=50):
    """
    Enumerate all solutions to the knapsack problem with the given counts and elements.
    Faster than ip_solve because it can be cached.
    """
    return ip_solve(counts, {e: 1 for e in elements}, num_solutions=num_solutions)

def ip_solve(counts: tuple, raprank: dict, dist: dict, n=1, num_solutions=50):
    ordering = get_ordering(dist)
    # constraint_mat = np.array(ordering).T
    constraint_mat = np.tile(np.identity(len(counts)), n)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model('IP solver', env=env)
    m.Params.LogToConsole = 0
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = num_solutions
    # var_dict = {i: m.addVar(vtype=GRB.INTEGER, name="x" + str(i)) for i, house in enumerate(ordering) if is_eligible(house, counts)}
    # var_list = np.array([var_dict[i] for i in range(len(var_dict))])
    x = m.addMVar(len(counts)*n, vtype=GRB.INTEGER, lb=0)
    # nl_probs = np.array([-log(dist[i]) for i in ordering])
    counts = torch.Tensor.cpu(counts)
    nl_probs = np.tile(np.array([dist[i] for i in ordering]), 114)
    # prob of sequence + approx multinomial correction
    # m.setObjective(nl_probs @ x + np.ones((len(ordering),)) @ x)
    # m.setObjective(nl_probs @ x, GRB.MINIMIZE)
    m.setObjective(0)
    m.addConstr(constraint_mat @ x== np.array(counts))
    i = 0
    for answer in raprank:
        # a: number of times the answer appears
        a = raprank[answer]
        # d: number of queries
        # n: number of households in the block
        d = len(counts)

        # Create binary decision variables z[i] (to indicate if row i matches X)
        z = m.addVars(n, vtype=GRB.BINARY, name="z")

        # Big M constant (since Y_flat and x are binary, M=1 is appropriate)
        M = 1


        # Add constraints to check if each row in the flattened Y matches X
        for i in range(n):
            # For each row i, check all d elements
            for j in range(d):
                # Y_flat[i * d + j] corresponds to the element in row i and column j
                m.addConstr((1 - z[i]) * M >= (x[i * d + j] - answer[j]), name=f"match_row_{i}_col_{j}_1")
                m.addConstr((1 - z[i]) * M >= (answer[j] - x[i * d + j]), name=f"match_row_{i}_col_{j}_2")

        # Constraint: Exactly 12 rows must match X
        m.addConstr(gp.quicksum(z[i] for i in range(n)) == a, name="match_count")
        # for j in range(n):
        #     m.addConstr(x[i:i + l]== np.array(torch.Tensor.cpu(answer)))
        #     i = i + l

        # break

    m.optimize()
    # for v in m.getVars():
        # print('%s %g' % (v.varName, v.x))
    nSolutions = m.SolCount
    print(f"n:{nSolutions}")

    # np.array(m.Xn).reshape(n, d) 

    # pdb.set_trace()
    values = m.Xn
    values = values[:-len(raprank) * n]
    values = np.array(values).reshape(n, d) 

    for answer, a in raprank.items():
        check = (values == answer.cpu().numpy()).all(1).sum() == a
        print(check, a)

    pdb.set_trace() 

    print('Obj: %g' % m.objVal)
    # print('Number of solutions found: ' + str(nSolutions))
    # for sol in range(nSolutions):
        # m.setParam(GRB.Param.SolutionNumber, sol)
        # print(m.PoolObjVal)
        # values = m.Xn
        # for v, h in zip(values, ordering):
            # if v > 0:
                # print(h, ':', int(v))

    sols = []
    for sol in range(nSolutions):
        m.setParam(GRB.Param.SolutionNumber, sol)
        values = m.Xn
        # current_sol = []
        # for v, h in zip(values, ordering):
        #     # if v > 0:
        #     current_sol += [h] * round(v)
        # current_sol = tuple(current_sol)
        # sols.append(current_sol)
        pdb.set_trace()
        assert(np.all(constraint_mat @ values == np.array(counts)))
        sols.append(values)
        print(f"solution: {values}")
    env.dispose()
    exit(0)
    return sols

if __name__ == '__main__':
    dist = {
            (1, 0, 1): 1,
            (0, 1, 1): 1,
            (1, 1, 1): 1,
            (2, 0, 1): 1,
            }
    dist = normalize(dist)
    sol = ip_solve((5, 1, 5), dist, num_solutions=3)
    print(sol)
