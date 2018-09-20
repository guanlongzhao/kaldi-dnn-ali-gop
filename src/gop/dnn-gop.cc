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
            std::string &model_in_filename,
            std::string &lex_in_filename) {
  bool binary;
  Input ki(model_in_filename, &binary);
  tm_.Read(ki.Stream(), binary);
  am_.Read(ki.Stream(), binary);
  ReadKaldiObject(tree_in_filename, &ctx_dep_);

  fst::VectorFst<fst::StdArc> *lex_fst = fst::ReadFstKaldi(lex_in_filename);
  std::vector<int32> disambig_syms;  
  TrainingGraphCompilerOptions gopts;
  gc_ = new TrainingGraphCompiler(tm_, ctx_dep_, lex_fst, disambig_syms, gopts);

  for (size_t i = 0; i < tm_.NumTransitionIds(); i++) {
    pdfid_to_tid[tm_.TransitionIdToPdf(i)] = i;
  }
}

BaseFloat DnnGop::Decode(fst::VectorFst<fst::StdArc> &fst,
                         nnet2::DecodableAmNnet &decodable,
                         std::vector<int32> *align) {
  FasterDecoderOptions decode_opts;
  decode_opts.beam = 500; // number of beams for decoding. Larger, slower and more successful alignments. Smaller, more unsuccessful alignments.
  FasterDecoder decoder(fst, decode_opts);
  decoder.Decode(&decodable);
  if (! decoder.ReachedFinal()) {
    KALDI_WARN << "Did not successfully decode.";
  }
  fst::VectorFst<LatticeArc> decoded;
  decoder.GetBestPath(&decoded);
  std::vector<int32> osymbols;
  LatticeWeight weight;
  GetLinearSymbolSequence(decoded, align, &osymbols, &weight);
  BaseFloat likelihood = -(weight.Value1()+weight.Value2());

  return likelihood;
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

  // Vector<BaseFloat> likelihood(phone_syms.size());

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
    // likelihood(i) = phn_likelihood;
  }

  return likelihood;
}

void DnnGop::GetContextFromSplit(std::vector<std::vector<int32> > split,
                                 int32 index, int32 &phone_l, int32 &phone, int32 &phone_r) {
  KALDI_ASSERT(index < split.size());
  phone_l = (index > 0) ? tm_.TransitionIdToPhone(split[index-1][0]) : 1;
  phone = tm_.TransitionIdToPhone(split[index][0]);
  phone_r = (index < split.size() - 1) ? tm_.TransitionIdToPhone(split[index+1][0]): 1;
}

void DnnGop::Compute(const CuMatrix<BaseFloat> &feats,
                     const std::vector<int32> &transcript) {
  // Align
  fst::VectorFst<fst::StdArc> ali_fst;
  gc_->CompileGraphFromText(transcript, &ali_fst);
  nnet2::DecodableAmNnet ali_decodable(tm_, am_, feats, true, 1.0);
  Decode(ali_fst, ali_decodable, &alignment_);
  KALDI_ASSERT(feats.NumRows() == alignment_.size());

  // GOP
  std::vector<std::vector<int32> > split;
  SplitToPhones(tm_, alignment_, &split);
  gop_result_.Resize(split.size());
  phones_.resize(split.size());
  phones_loglikelihood_.Resize(split.size());
  int32 frame_start_idx = 0;
  for (MatrixIndexT i = 0; i < split.size(); i++) {
    int32 phone, phone_l, phone_r;
    GetContextFromSplit(split, i, phone_l, phone, phone_r);

    BaseFloat gop_numerator = ComputeGopNumera(ali_decodable, phone_l, phone, phone_r, frame_start_idx, split[i].size());
    BaseFloat gop_denominator = ComputeGopDenomin(ali_decodable, phone_l, phone_r, frame_start_idx, split[i].size());
    gop_result_(i) = (gop_numerator - gop_denominator) / split[i].size();
    phones_[i] = phone;
    phones_loglikelihood_(i) = gop_numerator;

    frame_start_idx += split[i].size();
  }
}

Vector<BaseFloat>& DnnGop::Result() {
  return gop_result_;
}

Vector<BaseFloat>& DnnGop::get_phn_ll() {
  return phones_loglikelihood_;
}

std::vector<int32>& DnnGop::get_alignment() {
  return alignment_;
}

std::vector<int32>& DnnGop::Phonemes() {
  return phones_;
}

}  // End namespace kaldi
