// gop/dnn-gop.h

// Copyright 2016-2018  Junbo Zhang
//                      Ming Tu
//                      Guanlong Zhao

// This program is based on Kaldi (https://github.com/kaldi-asr/kaldi).
// However, this program is NOT UNDER THE SAME LICENSE of Kaldi's.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// version 2 as published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

#ifndef KALDI_GOP_DNN_H_
#define KALDI_GOP_DNN_H_ 1

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "decoder/training-graph-compiler.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "decoder/faster-decoder.h"
#include "fstext/fstext-utils.h"
#include "nnet2/decodable-am-nnet.h"
#include "matrix/matrix-common.h"

namespace kaldi {

class DnnGop {
public:
  DnnGop() {}
  void Init(std::string &tree_in_filename,
            std::string &model_in_filename);
  void Compute(const CuMatrix<BaseFloat> &feats,
               const std::vector<int32> &existing_phonemes,
               std::vector<int32> &num_frames);
  Vector<BaseFloat>& Result();

protected:
  nnet2::AmNnet am_;
  TransitionModel tm_;
  ContextDependency ctx_dep_;
  std::map<int32, int32> pdfid_to_tid;
  Vector<BaseFloat> gop_result_;               
  BaseFloat ComputeGopNumera(nnet2::DecodableAmNnet &decodable,
                                    int32 phone_l, int32 phone, int32 phone_r,
                                    MatrixIndexT start_frame,
                                    int32 size);
  BaseFloat ComputeGopDenomin(nnet2::DecodableAmNnet &decodable,
                              int32 phone_l, int32 phone_r,
                              MatrixIndexT start_frame,
                              int32 size);
};

}  // End namespace kaldi

#endif  // KALDI_GOP_DNN_H_
