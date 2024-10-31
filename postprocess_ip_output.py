import pandas as pd
from pathlib import Path
import os

rootdir = Path(os.getcwd())
file_list = [f for f in rootdir.glob('./ip_output/**/*') if f.is_file()]
output = []

for file in file_list: 
    # Load the CSV file
    df = pd.read_csv(file)

    # Separate the last three columns
    n_incorrect_correct = df.iloc[:, -3:]

    # Concatenate the rest of the columns into a string
    feature_vals = ['_'.join(row.astype(str)) for row in df.values[:, :-3]]
    features = '_'.join(df.columns[:-3])

    # Create a new DataFrame with the last three columns and the concatenated features
    for i in range(len(df)):
        row = {
            'features': features,
            'feature_vals': feature_vals[i],
            'pred': n_incorrect_correct.values[i, 0],
            'incorrect': n_incorrect_correct.values[i, 1],
            'correct': n_incorrect_correct.values[i, 2],
        }
        output.append(row)

new_df = pd.DataFrame(output)
new_df.to_csv('241030.csv', index=False)
