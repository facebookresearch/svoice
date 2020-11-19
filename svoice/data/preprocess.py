# The following piece of code was adapted from https://github.com/kaituoxu/Conv-TasNet
# released under the MIT License.
# Author: Kaituo XU
# Created on 2018/12

# Revised by: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf

import argparse
import json
import os

import librosa
from tqdm import tqdm


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        for signal in ['noisy', 'clean']:
            preprocess_one_dir(os.path.join(args.in_dir, data_type, signal),
                               os.path.join(args.out_dir, data_type),
                               signal,
                               sample_rate=args.sample_rate)


def preprocess_alldirs(args):
    for d in os.listdir(args.in_dir):
        local_dir = os.path.join(args.in_dir, d)
        if os.path.isdir(local_dir):
            preprocess_one_dir(os.path.join(args.in_dir, local_dir),
                               os.path.join(args.out_dir),
                               d,
                               sample_rate=args.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSJ0 data preprocessing")
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Directory path of wsj0 including tr, cv and tt')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate of audio file')
    parser.add_argument("--one_dir", action="store_true",
                        help="Generate json files from specific directory")
    parser.add_argument("--all_dirs", action="store_true",
                        help="Generate json files from all dirs in specific directory")
    parser.add_argument('--json_name', type=str, default=None,
                        help='The name of the json to be generated. '
                             'To be used only with one-dir option.')
    args = parser.parse_args()
    print(args)
    if args.all_dirs:
        preprocess_alldirs(args)
    elif args.one_dir:
        preprocess_one_dir(args.in_dir, args.out_dir,
                           args.json_name, sample_rate=args.sample_rate)
    else:
        preprocess(args)
