# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

DATE=`date +%d_%m_%Y`
CODE=src
OUTPUT=results/${DATE}/$1

echo "Sampling"
python ${CODE}/data_samples.py --data-from-args checkpoints/$1/args.pth --output ${OUTPUT}-py  -n 2 --seq 80000

echo "Generating"
python ${CODE}/run_on_files.py --files ${OUTPUT}-py --batch-size 2 --checkpoint checkpoints/$1/lastmodel --output-next-to-orig --decoders $2 --py
