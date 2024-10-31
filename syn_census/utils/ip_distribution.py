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

def ip_solve(counts: tuple, raprank: dict, dist: dict, n=1, num_solutions=500, constraint_flag = False):
    ordering = get_ordering(dist)
    constraint_mat = np.tile(np.identity(len(counts)), n)
    col_arr = []
    sols = []
    for answer in raprank:
        i = 0
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
        if constraint_flag:
            # Constraint: Exactly 12 rows must match X
            m.addConstr(gp.quicksum(z[i] for i in range(n)) == a, name="match_count")
        else:
            # Add a binary variable
            delta = m.addVar(vtype=GRB.BINARY, name="delta")
            # Add constraints
            m.addConstr((1-delta)*(gp.quicksum(z[i] for i in range(n))- a - 1) >= 0, name="match_count_upper")
            m.addConstr((delta)*(gp.quicksum(z[i] for i in range(n))- a + 1) <= 0, name="match_count_lower")
                #gp.quicksum(z[i] for i in range(n)) <= a - M * delta
            #m.addConstr(gp.quicksum(z[i] for i in range(n)) >= a + 1 - M * (1 - delta), name="match_count_upper")

        m.optimize()
        # for v in m.getVars():
            # print('%s %g' % (v.varName, v.x))
        nSolutions = m.SolCount
        print(f"n:{nSolutions}")
        
        if nSolutions > 0:
            # values = m.Xn
            # if not constraint_flag:
            #     values = values[:-len(raprank) * n]
            # values = np.array(values).reshape(n, d)
            col_arr.append(0)

            # for answer, a in raprank.items():
            #     check = (values == answer.cpu().numpy()).all(1).sum() == a
            #     assert(check)

            print('Obj: %g' % m.objVal)
        else:
            col_arr.append(1)
    # print('Number of solutions found: ' + str(nSolutions))
    # for sol in range(nSolutions):
        # m.setParam(GRB.Param.SolutionNumber, sol)
        # print(m.PoolObjVal)
        # values = m.Xn
        # for v, h in zip(values, ordering):
            # if v > 0:
                # print(h, ':', int(v))

        for sol in range(nSolutions):
            m.setParam(GRB.Param.SolutionNumber, sol)
            values = m.Xn
            values = values[:-len(raprank)]
            # assert(np.all(constraint_mat @ values == np.array(counts)))
            sols.append(values)
            # print(f"solution: {values}")
        env.dispose()
    return sols, col_arr

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
