
# kaldi-dnn-ali-gop
Computes forced-alignment and GOP (Goodness of Pronunciation) bases on Kaldi with nnet2 support. Can optionally output the phoneme confusion matrix on frame or phoneme segment level. The acoustic model is trained using librispeech database (960 hours data) with the scripts under kaldi/egs/librispeech. The GOP implementation is GOP<sub>1</sub> from [Witt and Young (2000)](http://hstrik.ruhosting.nl/wordpress/wp-content/uploads/2013/03/Witt-Young-2000-SpeCom30.pdf).

## Requirements
1. textgrid (https://github.com/kylebgorman/textgrid)
2. numpy (optional, only if you want the phoneme confusion measurements)

## How to build
1. Make sure you have installed [Kaldi](https://github.com/kaldi-asr/kaldi).
2. Make sure that you have installed the CUDA library.
3. Download this repo.
4. Change `KALDI_ROOT` in `src/CMakeLists.txt` to your Kaldi dir.
5. `cd src && mkdir build`, `cd build`, `cmake .. && make`
6. Add `src/build` to `PATH`.
7. Change `KALDI_ROOT` in `egs/gop-compute/path.sh` to your own `KALDI_ROOT`
8. Download prebuilt [AM and LM](https://drive.google.com/file/d/19SHvdARrzIbTuqF0SAqiV_0eRQuEmWLr/view?usp=sharing), unzip, and put folders `am` and `lm` in `egs/gop-compute`.

## Run the example
```
cd egs/gop-compute
./run.sh --nj 1 test_data/aba data/test exp/test
```

- --nj: number of jobs to do parallel computing, should be smaller than your number of CPU cores
- test_data/aba: sample data
- data/test: intermediate folder where acoustic features related files are stored
- exp/test: where results are stored (aligned_textgrid: alignments in textgrid format; gop/gop.txt: gop values for every phoneme of every utterance)

### Notes on data preparation
To use this tool, audio files (.wav) and corresponding transcript (.lab) needs to be prepared and stored in following format:

```
.
├── ...
├── speaker # indicate speaker ID
│   ├── recordings
|       ├── utt1.wav # indicate utterance ID
|       ├── utt2.wav 
│   ├── labs
|       ├── utt1.lab # indicate utterance ID
|       ├── utt2.lab 
│   ├── spk2gender # speaker m/f
└── ...
```

Do not use space in speaker folder name or utterance file name, using underscore instead. Please refer to Kaldi's documentation on [data preparation](http://kaldi-asr.org/doc/data_prep.html).