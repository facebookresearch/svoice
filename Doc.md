# Build docker
```
./docker/build.sh
```

# Run docker
```
./docker/run.sh
```

# Train
```
./make_debug.sh
python3 train.py
```

# Inference
```
python -m svoice.separate outputs/exp_/checkpoint.th output --mix_json egs/debug/tr/mix.json
```