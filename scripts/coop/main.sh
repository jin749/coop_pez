#!/bin/bash

# custom config
DATA=/hdd/hdd3/jsh/DATA
TRAINER=CoOp

DATASET=dtd
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=True  # class-specific context (False or True)

# for SEED in 1 2 3
# do
#     DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
#     if [ -d "$DIR" ]; then
#         echo "Oops! The results exist at ${DIR} (so skip this job)"
#     else
#         python train.py \
#         --root ${DATA} \
#         --seed ${SEED} \
#         --trainer ${TRAINER} \
#         --dataset-config-file configs/datasets/${DATASET}.yaml \
#         --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#         --output-dir ${DIR} \
#         TRAINER.COOP.N_CTX ${NCTX} \
#         TRAINER.COOP.CSC ${CSC} \
#         TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
#         DATASET.NUM_SHOTS ${SHOTS}
#     fi
# done


SEED=1

DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
