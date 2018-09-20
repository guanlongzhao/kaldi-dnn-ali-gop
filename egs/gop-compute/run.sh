#!/bin/bash

# Copyright 2017   Author: Ming Tu
# Modified  2018   Author: Guanlong Zhao
# Arguments:
# input-dir: where input files are stored
# data-dir: where extracted features are stored
# result-dir: where results are stored                               

nj=1 # number of parallel jobs. Set it to number of CPU cores

# Enviroment preparation
. ./cmd.sh
. ./path.sh
[ -h steps ] || ln -s $KALDI_ROOT/egs/wsj/s5/steps
[ -h utils ] || ln -s $KALDI_ROOT/egs/wsj/s5/utils
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: run.sh <input-dir> <data-dir> <result-dir>"
   echo "  --nj                                  # number of jobs"
   exit 1;
fi

input_dir=$1
data_dir=$2
result_dir=$3

# Data preparation
local/data_prep.sh --nj $nj $input_dir $data_dir

# Calculation
local/compute-dnn-gop.sh --nj "$nj" --cmd "$decode_cmd" $data_dir lm/ am/ $result_dir

# Test
# ./run.sh --nj 1 test_data/aba data/test exp/test
