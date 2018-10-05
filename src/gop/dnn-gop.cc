// gop/dnn-gop.cc

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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-utils.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "lat/kaldi-lattice.h"
#include "hmm/hmm-utils.h"
#include "gop/dnn-gop.h"

namespace kaldi {

typedef typename fst::StdArc Arc;
typedef typename Arc::StateId StateId;
typedef typename Arc::Weight Weight;

void DnnGop::Init(std::string &tree_in_filename,
            std::string &model_in_filename) {
  bool binary;
  Input ki(model_in_filename, &binary);
  tm_.Read(ki.Stream(), binary);
  am_.Read(ki.Stream(), binary);
  ReadKaldiObject(tree_in_filename, &ctx_dep_);

  // Technically this mapping is not correct, because multiple transition IDs
  // can use the same pdf ID, so in the end this for loop will map each pdf ID
  // to the last transition ID it encounters. However, in this specific program,
  // this is okay because we do not really need the actual transition ID for
  // each frame when decoding using a nnet2 model, see comments where
  // pdfid_to_tid is referenced.
  for (size_t i = 0; i < tm_.NumTransitionIds(); i++) {
    pdfid_to_tid[tm_.TransitionIdToPdf(i)] = i;
  }
}

BaseFloat DnnGop::ComputeGopNumera(nnet2::DecodableAmNnet &decodable,
                                          int32 phone_l, int32 phone, int32 phone_r,
                                          MatrixIndexT start_frame,
                                          int32 size) {
  KALDI_ASSERT(ctx_dep_.ContextWidth() == 3);
  KALDI_ASSERT(ctx_dep_.CentralPosition() == 1);
  std::vector<int32> phoneseq(3);
  phoneseq[0] = phone_l;
  phoneseq[1] = phone;
  phoneseq[2] = phone_r;

  const int pdfclass_num = tm_.GetTopo().NumPdfClasses(phone);

  BaseFloat likelihood = 0;
  for (MatrixIndexT frame = start_frame; frame < start_frame + size; frame++) {
    Vector<BaseFloat> temp_likelihood(pdfclass_num);
    for (size_t c = 0; c < pdfclass_num; c++) {
      int32 pdf_id;
      if (!ctx_dep_.Compute(phoneseq, c, &pdf_id)) {
        KALDI_ERR << "Failed to obtain pdf_id";
      }

      // As the comment at where pdfid_to_tid is initialized suggested, this tid
      // may not be the real tid of the current frame, because the mapping from
      // pdf id to tid is in fact one-to-many. However, when obtaining the
      // log likelihood from a nnet2 decoding, this does not matter, and here is
      // the explanation. decodable.LogLikelihood(frame, tid) uses tid as one of
      // its inputs, and essentially what this function does is to get the value
      // of log_probs_(frame, pdf_id) (kaldi's decodable-am-nnet.h), so what
      // LogLikelihood() does is to convert tid to the pdf ID it is mapped to
      // through trans_model_.TransitionIdToPdf(tid), and then calls log_probs_.
      // This kaldi implementation looks really weird but I guess it is due to
      // some legacy dependencies.
      // To sum up, the value of tid is not correct here, but it does not
      // matter, it is used to workaroud a kaldi ugliness.
      int32 tid = pdfid_to_tid[pdf_id];

      // The LogLikelihood() is not a probability -- it is P(s|o)/P(s), where s
      // is an output state (senone) and o is the acoustic observation. Note
      // that the real numerator in GOP_1 is P(o|s)=P(s|o)P(o)/P(s), but since
      // the denominator also has P(o) so we can get rid of the acoustic feature
      // priors. This comment applies to ComputeGopDenomin() too. An issue
      // caused by this is that the "log likelihood" can be greater than 0, so
      // when you see a large positive "log likelihood", keep in mind that it is
      // not a probablity.
      temp_likelihood(c) = decodable.LogLikelihood(frame, tid);
    }
    likelihood += temp_likelihood.LogSumExp(5);
  }

  return likelihood;
}

BaseFloat DnnGop::ComputeGopDenomin(nnet2::DecodableAmNnet &decodable,
                                    int32 phone_l, int32 phone_r,
                                    MatrixIndexT start_frame,
                                    int32 size) {
  KALDI_ASSERT(ctx_dep_.ContextWidth() == 3);
  KALDI_ASSERT(ctx_dep_.CentralPosition() == 1);
  std::vector<int32> phoneseq(3);
  phoneseq[0] = phone_l;
  phoneseq[2] = phone_r;

  BaseFloat likelihood = -10000000;

  const std::vector<int32> &phone_syms = tm_.GetPhones();

  for (size_t i = 0; i < phone_syms.size(); i++) {
    int32 phone = phone_syms[i];
    phoneseq[1] = phone;
    const int pdfclass_num = tm_.GetTopo().NumPdfClasses(phone);
    
    BaseFloat phn_likelihood = 0;
    for (MatrixIndexT frame = start_frame; frame < start_frame + size; frame++) {
      Vector<BaseFloat> temp_likelihood(pdfclass_num);
      for (size_t c = 0; c < pdfclass_num; c++) {
        int32 pdf_id;
        if (!ctx_dep_.Compute(phoneseq, c, &pdf_id)) {
          KALDI_ERR << "Failed to obtain pdf_id";
        }
        int32 tid = pdfid_to_tid[pdf_id];

        temp_likelihood(c) = decodable.LogLikelihood(frame, tid); 
      }
      phn_likelihood += temp_likelihood.LogSumExp(5);
    }
    // Take the maximum of the log likelihood among all phonemes.
    // This is the GOP_1 that Witt & Young (2000) originally proposed.
    if (phn_likelihood > likelihood) {
      likelihood = phn_likelihood;
    }
  }

  return likelihood;
}

void DnnGop::Compute(const CuMatrix<BaseFloat> &feats,
             const std::vector<int32> &existing_phonemes,
             std::vector<int32> &num_frames) {
  // Compute logprob and do some sanity check
  nnet2::DecodableAmNnet ali_decodable(tm_, am_, feats, true, 1.0);
  KALDI_ASSERT(existing_phonemes.size() == num_frames.size());
  int32 num_phones = existing_phonemes.size();

  // Deal with frame mismatch issue
  int32 total_num_frames = 0;
  for (auto& n : num_frames) {
    total_num_frames += n;
  }
  int32 num_diff = feats.NumRows() - total_num_frames;
  if (num_diff > 0) {
    KALDI_LOG << "Append " << num_diff << " frames to the end of the input.";
    num_frames.back() += num_diff;
  }
  else {
    if (num_diff < 0) {
      KALDI_LOG << "Remove " << -num_diff << " frames from the end of the input.";
      int32 last_valid_chunk_idx = num_phones - 1;
      while (last_valid_chunk_idx >= 0 && num_diff < 0) {
        if (num_frames[last_valid_chunk_idx] >= (-num_diff)) {
          // have enough frames to remove, remove and stop
          num_frames[last_valid_chunk_idx] += num_diff;
          num_diff = 0;
        }
        else {
          // not enough, remove as much as possible
          num_diff += num_frames[last_valid_chunk_idx];
          num_frames[last_valid_chunk_idx] = 0;
        }
        last_valid_chunk_idx -= 1;
      }
    }
  }
  total_num_frames = 0;
  for (auto& n : num_frames) {
    total_num_frames += n;
  }
  KALDI_ASSERT(feats.NumRows() == total_num_frames);

  // GOP
  gop_result_.Resize(existing_phonemes.size());
  int32 frame_start_idx = 0;
  for (MatrixIndexT i = 0; i < num_phones; i++) {
    int32 phone, phone_l, phone_r;
    phone_l = (i > 0) ? existing_phonemes[i-1] : 1;
    phone = existing_phonemes[i];
    phone_r = (i < num_phones - 1) ? existing_phonemes[i+1] : 1;

    if (num_frames[i] > 0) {
      BaseFloat gop_numerator = ComputeGopNumera(ali_decodable, phone_l, phone, phone_r, frame_start_idx, num_frames[i]);
      BaseFloat gop_denominator = ComputeGopDenomin(ali_decodable, phone_l, phone_r, frame_start_idx, num_frames[i]);
      gop_result_(i) = (gop_numerator - gop_denominator) / num_frames[i];
    }
    else {
      gop_result_(i) = 0;
    }
    frame_start_idx += num_frames[i];
  }
}

Vector<BaseFloat>& DnnGop::Result() {
  return gop_result_;
}
}  // End namespace kaldi
