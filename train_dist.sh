# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

set -e -x

CODE=src
DATA=musicnet/preprocessed
EXP=musicnet_dist
MASTER_PORT=29500

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --nnodes=$1 \
        --node_rank=$2 \
        --master_addr=$3 \
        --master_port=${MASTER_PORT} \
    ${CODE}/train.py \
        --data ${DATA}/Bach_Solo_Cello  \
               ${DATA}/Beethoven_Solo_Piano \
               ${DATA}/Cambini_Wind_Quintet \
               ${DATA}/Bach_Solo_Piano \
               ${DATA}/Beethoven_Accompanied_Violin \
               ${DATA}/Beethoven_String_Quartet  \
        --batch-size 32 \
        --lr-decay 0.995 \
        --epoch-len 1000 \
        --num-workers 5 \
        --lr 1e-3 \
        --seq-len 12000 \
        --d-lambda 1e-2 \
        --expName ${EXP} \
        --latent-d 64 \
        --layers 14 \
        --blocks 4 \
        --data-aug \
        --grad-clip 1