// gopbin/compute-gop-dnn.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gop/dnn-gop.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    
    const char *usage =
        "Compute GOP with DNN-based models.\n"
        "Usage:   compute-dnn-gop [options] tree-in model-in feature-rspecifier "
        "gop-wspecifier phoneme-rspecifier numframe-rspecifier\n"
        "e.g.: \n"
        " compute-dnn-gop tree 1.mdl scp:train.scp ark,t:gop.1 ark:phoneme.txt ark:numframes.txt\n";

    ParseOptions po(usage);
    std::string use_gpu = "no";
    po.Register("use-gpu", &use_gpu,
                 "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);
    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string tree_in_filename = po.GetArg(1);
    std::string model_in_filename = po.GetArg(2);
    std::string feature_rspecifier = po.GetArg(3);
    std::string gop_wspecifier = po.GetArg(4);
    std::string phoneme_rspecifier = po.GetArg(5);
    std::string frame_rspecifier = po.GetArg(6);

    SequentialBaseFloatCuMatrixReader feature_reader(feature_rspecifier);
    BaseFloatVectorWriter gop_writer(gop_wspecifier);
    RandomAccessInt32VectorReader phoneme_reader(phoneme_rspecifier);
    RandomAccessInt32VectorReader frame_reader(frame_rspecifier);

    DnnGop gop;
    gop.Init(tree_in_filename, model_in_filename);

    // Compute for each utterance
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!(phoneme_reader.HasKey(utt) && frame_reader.HasKey(utt))) {
        KALDI_WARN << "Can not find alignment for utterance " << utt;        
        continue;
      }

      KALDI_LOG << "Processing utterance " << utt;

      const CuMatrix<BaseFloat> &features = feature_reader.Value();
      const std::vector<int32> &existing_phonemes = phoneme_reader.Value(utt);
      std::vector<int32> num_frames = frame_reader.Value(utt);
      std::vector<int32> &num_frames_var = num_frames;

      gop.Compute(features, existing_phonemes, num_frames_var);
      gop_writer.Write(utt, gop.Result());
    }
    KALDI_LOG << "Done.";
# if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
