set -x

CONFIG=$1
CKPT=$2
BATCH=${3:-"32"}
GPUS=${4:-"0"}

python ./scripts/validate.py \
    --cfg ${CONFIG} \
    --batch ${BATCH} \
    --gpus $GPUS\
    --flip-test \
    --checkpoint ${CKPT}
