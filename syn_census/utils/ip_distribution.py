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
        x = m.addMVar(len(counts)*n, vtype=GRB.BINARY, lb=0)
        # nl_probs = np.array([-log(dist[i]) for i in ordering])
        counts = torch.Tensor.cpu(counts)
        nl_probs = np.tile(np.array([dist[i] for i in ordering]), 114)
        # prob of sequence + approx multinomial correction
        # m.setObjective(nl_probs @ x + np.ones((len(ordering),)) @ x)
        # m.setObjective(nl_probs @ x, GRB.MINIMIZE)
        m.setObjective(0)
        m.addConstr(constraint_mat @ x == np.array(counts))
        # a: number of times the answer appears
        a = raprank[answer]
        # d: number of queries
        # n: number of households in the block
        d = len(counts)

        # Create binary decision variables z[i] (to indicate if row i matches X)
        z = m.addVars(n, vtype=GRB.BINARY, name="z")
        y = m.addVars(n, vtype=GRB.BINARY, name="y")

        # Big M constant (since Y_flat and x are binary, M=1 is appropriate)
        M = n * 2

        w = m.addVars(constraint_mat.shape[1], vtype=GRB.BINARY, name="w")
        v = m.addVars(constraint_mat.shape[1], vtype=GRB.BINARY, name='v')
        for i in range(n):
            for j in range(d):
                idx = i * d + j

                m.addConstr(x[idx] - answer[j] <= M * (1 - w[idx])) # 1
                m.addConstr(answer[j] - x[idx] <= M * (1 - w[idx])) # 2
                m.addConstr(x[idx] - answer[j] >= 1 - M * w[idx] - (1 - w[idx]) * M * v[idx]) # 3
                m.addConstr(answer[j] - x[idx] >= 1 - M * w[idx] - (1 - w[idx]) * M * (1 - v[idx])) # 4

                """
                w = 0:
                    1) x - a <= M
                    2) a - x <= M
                        --> |x - a| <= M --> anything goes
                    b = 0:
                        3) x - a >= 1
                        4) a - x >= 1 - M --> x - a <= M-1
                            --> 1 <= x-a <= M-1
                    b = 1
                        3) x - a >= 1 - M
                        4) a - x >= 1 --> x - a <= -1
                            --> 1 - M <= x - a <= -1
                w = 1 (b term doesn't matter when w = 1)
                    1) x - a <= 0
                    2) a - x <= 0
                        --> x - a == 0
                    3) x - a >= 1 - M
                    4) a - x >= 1 - M --> x - a <= M - 1
                        --> anything goes
                """

            # if w matches answer at all indices, then it should be all 1's --> sum 
            w_sum = gp.quicksum(w[k] for k in range(i * d, (i + 1) * d))
            # logic same as above for making z an indicator of w_sum == d
            m.addConstr(w_sum - d <= M * (1 - z[i]))
            m.addConstr(d - w_sum <= M * (1 - z[i]))
            m.addConstr(w_sum - d >= 1 - M * z[i] - (1 - z[i]) * M * y[i])
            m.addConstr(d - w_sum >= 1 - M * z[i] - (1 - z[i]) * M * (1 - y[i]))

        z_sum = gp.quicksum(z[i] for i in range(n))

        if constraint_flag:
            m.addConstr(z_sum == a, name="match_count")
        else:
            check = m.addVar(vtype=GRB.BINARY, name="delta")
            m.addConstr(z_sum - a <= M * check - 1, name="match_count_upper")
            m.addConstr(z_sum - a >= 1 - M * (1 - check), name="match_count_lower")

            """
            check = 0:
                1) z_sum - a <= M - 1
                2) z_sum - a >= 1
                    --> 1 <= z_sum - a <= M - 1 --> z_sum - a != 0
            check = 1
                1) z_sum - a <= -1
                2) z_sum - a >= 1 - M
                    --> 1 - M <= z_sum - a <= -1 --> z_sum - a != 0
            """

        
        m.optimize()
        # for v in m.getVars():
            # print('%s %g' % (v.varName, v.x))
        nSolutions = m.SolCount
        # print(f"n:{nSolutions}")
        
        if nSolutions > 0:
            # values = m.Xn
            # if not constraint_flag:
            #     values = values[:-len(raprank) * n]
            # values = np.array(values).reshape(n, d)
            col_arr.append(0)

            # for answer, a in raprank.items():
            #     check = (values == answer.cpu().numpy()).all(1).sum() == a
            #     assert(check)

            # print('Obj: %g' % m.objVal)
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
            values = values[:x.shape[0]]
            # assert(np.all(constraint_mat @ values == np.array(counts)))
            sols.append(values)
            # print(f"solution: {values}")

        # answer.cpu().numpy() == np.array(values)[:x.shape[0]].reshape((n, -1))[0]

        # for i in range(n):
        #     for j in range(d):
        #         print((values[i * d + j] - answer[j]))

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
