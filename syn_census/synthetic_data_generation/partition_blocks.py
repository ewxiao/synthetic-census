import pandas as pd
import os
import pickle as pkl
import sys
import numpy as np
import torch
import pdb
from pathlib import Path
from .guided_solver import SOLVER_PARAMS, SOLVER_RESULTS, SolverResults, solve, reduce_dist
from ..utils.encoding import encode4_row, encode4_hh_dist, encode_raprank
from ..utils.census_utils import *
from ..preprocessing.build_micro_dist import read_microdata
from .mcmc_sampler import MCMCSampler
from syndata_reconstruction.utils.general import get_qm
from dp_data import get_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def read_block_data(block_clean_file: str):
    return pd.read_csv(block_clean_file)

def sample_from_sol(sol):
    keys, probs = zip(*sol.items())
    keys = list(keys)
    if len(keys) > 1:
        return keys[np.random.choice(range(len(keys)), p=probs)]
    else:
        return keys[0]

def generate_data(
        micro_file: str,
        block_clean_file: str,
        num_sols: int,
        task: int,
        num_tasks: int,
        include_probs: bool = False,
        tmp_file: str= '',
        weights: dict = None,
        root_path: str = '',
        subdir: str = '',
        dataset: str = '',
        marginal: int = -1,
        feature_path: str = '' # e.g. "./raprank_output/${SUBDIR}/${GEOID}/${PARTITION}/init_random/queries_${MARGINAL}/RP/K_${K}-T_${T}/*.csv",
        ):
    SOLVER_PARAMS.num_sols = num_sols
    geo_id = dataset
    ip_output = f"./ip_output/{geo_id}"
    Path(ip_output).mkdir(parents=True, exist_ok=True)
    features = Path(feature_path).stem
    features_list = features.split("_")
    features_full = ["TEN", "VACS", "HHSIZE", "HHT", "HHT2", "CPLT", "UPART", "MULTG", "PAC", 
                "TP18", "TP60", "TP65", "TP75", "PAOC", "HHSEX", "THHSPAN", "THHRACE", "THHLDRAGE"]
    features_drop = [feat for feat in features_full if feat not in set(features_list)]
    data = get_dataset(geo_id, root_path=root_path)
    domain = data.domain
    n = len(data.df)
    query_manager = get_qm([marginal], geo_id, data, root_path, device, unit = True)
    idxs_keep = []
    unique_queries = []
    descriptions = [query_manager.get_query_desc(i) for i in range(query_manager.num_queries)]
    for idx, workload in enumerate(query_manager.workloads):
        contains_other_cols = np.any([col not in features_list for col in workload])
        if not contains_other_cols:
            print(workload)
            idxs_keep.append(idx)
    query_manager.filter_query_workloads(idxs_keep)
    answers = query_manager.get_answers(data, density = False)
    hh_dist = encode4_hh_dist(data, domain, query_manager)
    answer_encoding = encode4_row(answers)
    raprank_encoding = encode_raprank(feature_path, domain, query_manager)
    errors = []
    output = []
    if tmp_file and os.path.exists(tmp_file):
        print('Loading tmp file', tmp_file)
        try:
            with open(tmp_file, 'rb') as f:
                output, errors = pkl.load(f)
        except:
            print('Error loading tmp file', tmp_file)
    already_finished = set([o['id'] for o in output])

    samplers = {}
    orig_df = pd.read_csv(feature_path)

    sol,col_arr = solve(hh_dist, raprank_encoding, n = n,answers = answers, constraint_flag = True)
    x = len(sol)
    print(x, 'unique solutions')
    solution_path = feature_path[:-4] + "T.txt"
    # with open(solution_path, 'w') as f:
    #     f.write('\n'.join([' '.join(f"{x}") for x in sol]))
    orig_df['ip_incorrect'] = col_arr

    sol, col_arr = solve(hh_dist, raprank_encoding, n = n,answers = answers, constraint_flag = False)
    x = len(sol)
    print(x, 'unique solutions')
    solution_path = feature_path[:-4] + "_F.txt"
    # with open(solution_path, 'w') as f:
    #     f.write('\n'.join([' '.join(f"{x}") for x in sol]))
    orig_df['ip_correct'] = col_arr
    
    orig_df.to_csv(os.path.join(ip_output, f"{features}.csv"), index = False)

    # chosen = sample_from_sol(sol)
    # # If not all solutions have been found, use MCMC
    # if len(sol) == num_sols:
    #     print('Using MCMC')
    #     counts = answer_encoding
    #     level = SOLVER_RESULTS.level
    #     use_age = SOLVER_RESULTS.use_age
    #     solve_dist = hh_dist
    #     # if level > 1:
    #     #     solve_dist = reduce_dist(hh_dist, level, use_age)
    #     #     counts = counts.reduce(level, use_age)
    #     tag = (level, use_age)
    #     if tag not in samplers:
    #         #TODO: make num_iterations and k parameters
    #         samplers[tag] = MCMCSampler(solve_dist, num_iterations=10000, k=3, max_solutions=num_sols)
    #     sampler = samplers[tag]
    #     try:
    #         chosen = sampler.mcmc_solve(answer_encoding, chosen)
    #     except:
    #         print('Error in MCMC')
    # # chosen = tuple(hh.to_sol() for hh in chosen)
    # if hasattr(chosen[0], 'get_type'):
    #     chosen_types = tuple(c.get_type() for c in chosen)
    #     print(chosen_types)
    # else:
    #     chosen_types = None
    # if SOLVER_RESULTS.status == SolverResults.UNSOLVED:
    #     print(ind, SOLVER_RESULTS.status, file=sys.stderr)
    #     errors.append(ind)
    # print('SOLVER LEVEL', SOLVER_RESULTS.level, 'USED AGE', SOLVER_RESULTS.use_age, 'STATUS', SOLVER_RESULTS.status)
    # if len(sol) > 0:
    #     d = {
    #             'id': identifier,
    #             'sol': chosen,
    #             'level': SOLVER_RESULTS.level,
    #             'complete': SOLVER_RESULTS.status == SolverResults.OK,
    #             'age': SOLVER_RESULTS.use_age,
    #             'types': chosen_types,
    #             }
    #     if include_probs:
    #         d['prob_list'] = list(sol.values())
    #     output.append(d)
    #     if i > 0 and i % 10 == 0 and tmp_file:
    #         print('Saving tmp file', tmp_file)
    #         with open(tmp_file, 'wb') as f:
    #             pkl.dump((output, errors), f)
    # return (output, errors)
