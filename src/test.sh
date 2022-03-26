set -e

# select fold index
FOLD=1

# select ml model type
MODEL=rf

# number of trails
NUM_TRAILS=10

# Submit message
SUBMIT_MESSAGE="${MODEL} eval fold ${FOLD} trails ${NUM_TRAILS} using complete pipeline"

SECONDS=0

sleep 5

echo ${SECONDS}

SECONDS=0

sleep 3

echo ${SECONDS}

