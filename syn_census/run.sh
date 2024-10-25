#!/bin/bash

VERSION=2023-04-03

FILE_DIR=datasets/preprocessed/ppmf/unit/${VERSION}/geo_sets

FILES=()

# TRACT=42003140100 # CMU
# FILES+=(${FILE_DIR}/${TRACT}.txt)

FILES+=(${FILE_DIR}/block/quantile_0.50.txt)
# FILES+=(${FILE_DIR}/block/quantile_0.25.txt)
# FILES+=(${FILE_DIR}/block/quantile_0.75.txt)
# FILES+=(${FILE_DIR}/block/max.txt)

for FILE in "${FILES[@]}"; do
    echo $FILE
done

SUBDIR=ppmf

MARGINAL=-1

NUM_RUNS=25
T=5000
K=-1

for FILE in "${FILES[@]}"; do
    while IFS= read -r GEOID || [ -n "$GEOID" ]; do
        echo $GEOID
        STATE="${GEOID:0:2}"
        ROOT_PATH=./datasets/preprocessed/ppmf/unit/${VERSION}/${STATE}
        for FEATURE_FILE in ./raprank_output/${SUBDIR}/${GEOID}/full/init_random/queries_${MARGINAL}/RP/K_{K}-T_{T}; do
            FEATURES=$(basename $FEATURE_FILE .csv)

            python3 generate_data_shard.py --from_params PA_params.json --task 1 --num_tasks 300 --task_name test \
                    --root_path $ROOT_PATH --subdir $SUBDIR --dataset $GEOID --marginal $MARGINAL --features $FEATURES
            done
        done
    done <$FILE
done
