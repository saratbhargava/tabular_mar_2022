#/bin/bash

# exit immediately if command returns non-zero status
set -e

#########  PARAMTERS ######################

# select fold index
FOLD=2

# select feature extraction
FEATURES="power_transform"

# select ml model type
MODEL=rf

# tune the model
IS_TUNE=true # true

# number of trails
NUM_TRAILS=3

# Kaggle Submission
IS_SUBMIT=true # true
SUBMIT_MESSAGE="${MODEL} eval fold ${FOLD} trails ${NUM_TRAILS} using complete pipeline"

#########  TRAIN AND SUBMIT MODEL ###########

# Best Model
MODEL_FILENAME=${MODEL}_fold${FOLD}_`date +%m_%d_%Y_%H_%M_%S`.bin

SUBMIT_FILENAME=${MODEL_FILENAME:0:-4}.csv

echo Training ${MODEL} model started at `date`
SECONDS=0

# train the model
if ${IS_TUNE} ; then
    python train.py --fold ${FOLD} --model ${MODEL} --features "${FEATURES}" --tune --num_trails ${NUM_TRAILS} --model_filename ${MODEL_FILENAME}
else
    python train.py --fold ${FOLD} --model ${MODEL} --features "${FEATURES}" --model_filename ${MODEL_FILENAME}
fi

train_duration=${SECONDS}
echo Training ${MODEL} model complete at `date`
echo Training duration: ${train_duration} seconds
echo ""

# test the model
echo Inference ${MODEL} model started at `date`
SECONDS=0

python inference.py --model_filename ${MODEL_FILENAME} --submit_filename ${SUBMIT_FILENAME}

inference_duration=${SECONDS}
echo Inference ${MODEL} model complete at `date`
echo Inference duration: ${inference_duration} seconds
echo ""

# Submit the output csv to kaggle
if ${IS_SUBMIT} ; then
    kaggle competitions submit -c tabular-playground-series-feb-2022 -f ../submit/${SUBMIT_FILENAME} -m "${SUBMIT_MESSAGE}"
    echo Kaggle submission complete
fi

##############################################
