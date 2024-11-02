import os
import glob
import pandas as pd
from pprint import pprint

import pdb

# Replace 'your_directory_path' with the path to your directory
directory_path = 'ip_output/010030102001008'

cols = ['correct', 'ip_correct', 'ip_incorrect', 'no_bugs', 'ip_success']

results = []
for filepath in glob.glob(os.path.join(directory_path, '*.csv')):
    # print(filepath)

    df = pd.read_csv(filepath)

    # debugging
    df['no_bugs'] = True
    # if the candidate is correct, but IP says it is definitely incorrect, there is a bug
    issues = (df['correct']) & (df['ip_incorrect'])
    df['no_bugs'] &= ~issues
    # if the candidate is incorrect, but IP says it is definitely correct, there is a bug
    issues = (~df['correct']) & (df['ip_correct'])
    df['no_bugs'] &= ~issues
    # IP cannot say it is both definitely correct and incorrect
    issues = (df['ip_correct']) & (df['ip_incorrect'])
    df['no_bugs'] &= ~issues

    # check if IP was actually right about anything
    df['ip_success'] = False
    # if the candidate is correct, and IP says it is definitely correct
    successes = (df['correct']) & (df['ip_correct'])
    df['ip_success'] |= successes
    # if the candidate is incorrect, and IP says it is definitely incorrect
    successes = (~df['correct']) & (df['ip_incorrect'])
    df['ip_success'] |= successes

    df['success_and_not_in_tables'] = df['ip_success'] & ~df['in_tables']

    assert df['no_bugs'].all(), filepath
    # if not df['no_bugs'].all():
    #     print(filepath)

    results.append(df[cols])

results = pd.concat(results)

pprint(results.sum())