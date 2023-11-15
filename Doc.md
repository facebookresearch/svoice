# Build docker
```
./docker/build.sh
```

# Run docker
```
./docker/run.sh
```

# Train
* Create training data:
```
python3 scripts/make_mix.py
```
* Create configuration:
```
./make_debug.sh
```
* Run the training
```
python3 train.py
```

## Some comments
* The original paper was trained on [WSJ](https://github.com/fgnt/sms_wsj) dataset, which is not public, but I think it is easy to get it.
* Training on big data is crutial
* [Need for pretrained model](https://github.com/facebookresearch/svoice/issues/1)

# Inference
```
python -m svoice.separate outputs/exp_/checkpoint.th output --mix_json egs/debug/tr/mix.json
```

# Creating a new dataset
## LJSpeach dataset
* Extract dataset:
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2
```

* Get noise dataset:
```
curl -L -O https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
unzip wham_noise.zip -d datset/wham_noise
```

* Put single voices to `dataset/new_dataset/si' folders

*
```
python3 scripts/make_dataset.py --in_path dataset/new_dataset/tr --out_path dataset/new_out --noise_path dataset/wham_noise/wham_noise/cv
```


## Other solutions
* [Nvidia's QuartzNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/wsj_quartznet_15x5)


## Create WSJ-0
* https://github.com/fgnt/sms_wsj
* http://github.com/kaldi-asr/kaldi

Install kaldi:
```
cd kaldi
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../dist ..
cmake --build . --target install -- -j8
```

Install SMS_WSJ
```
cd sms_wsj
pip install --user -e ./
export KALDI_ROOT=/path/to/kaldi
make WSJ_DIR=/path/to/wsj SMS_WSJ_DIR=/path/to/write/db/to
```