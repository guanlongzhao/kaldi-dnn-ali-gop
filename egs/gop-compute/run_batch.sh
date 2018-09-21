#!/bin/bash

# Copyright 2018    Guanlong Zhao
# Arguments:
# input-dir: where speaker folders are stored                      

nj=1 # number of parallel jobs. Set it to number of CPU cores

# Enviroment preparation
. ./cmd.sh
. ./path.sh
[ -h steps ] || ln -s $KALDI_ROOT/egs/wsj/s5/steps
[ -h utils ] || ln -s $KALDI_ROOT/egs/wsj/s5/utils
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "usage: run.sh <input-dir>"
   echo "  --nj # number of jobs"
   exit 1;
fi

input_dir=$1

for speaker_dir in ${input_dir}/*; do
    speaker_name="$(basename $speaker_dir)"
    time_mark="$(date +%H%M%S_%m_%d_%Y)"
    echo "Processing speaker: ${speaker_name}"
    ./run.sh --nj $nj ${speaker_dir} data/${speaker_name}_${time_mark} exp/${speaker_name}_${time_mark}
done

# Test
# ./run_batch.sh --nj 1 test_data
