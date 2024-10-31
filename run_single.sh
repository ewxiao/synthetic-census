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

PARTITION=full

GEOID=010030102001008

echo $GEOID
STATE="${GEOID:0:2}"
ROOT_PATH=./datasets/preprocessed/ppmf/unit/${VERSION}/${STATE}
python ../syndata-reconstruction/run/train.py --root_path $ROOT_PATH --subdir $SUBDIR --dataset $GEOID --data_partition $PARTITION \
    --num_runs $NUM_RUNS --K $K --T $T --marginal $MARGINAL --unit;
for i in {1..9}; do
    python ../syndata-reconstruction/run/eval_multiplicity_simplified.py --root_path $ROOT_PATH --subdir $SUBDIR --dataset $GEOID --data_partition $PARTITION \
        --num_runs $NUM_RUNS --K $K --T $T --marginal $MARGINAL --num_datasets_sampled 1 --group_size $i --unit; 
done
for FEATURE_FILE in ./raprank_output/${SUBDIR}/${GEOID}/${PARTITION}/init_random/queries_${MARGINAL}/RP/K_${K}-T_${T}/*.csv; do
# for FEATURE_FILE in ./raprank_test/${GEOID}/*.csv; do
    FEATURES=$(basename "$FEATURE_FILE" .csv)
    # TODOs: postprocessing data shard outputs (later), function that filters queries from query manager, add flag for constraints (== and !=) and output if num solutions == 0, know it is incorrect or correct, 
    # function that checks what candidates can read off tables
    python3 generate_data_shard.py --from_params PA_params.json --task 1 --num_tasks 300 --task_name test \
            --root_path $ROOT_PATH --subdir $SUBDIR --dataset $GEOID --marginal $MARGINAL --feature_path $FEATURE_FILE
done
