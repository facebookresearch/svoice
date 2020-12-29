# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Yossi Adi (adiyoss)

import argparse
import logging
import os
import sys

import librosa
import torch
import tqdm

from .data.data import EvalDataLoader, EvalDataset
from . import distrib
from .utils import remove_pad

from .utils import bold, deserialize_model, LogProgress
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser("Speech separation using MulCat blocks")
parser.add_argument("model_path", type=str, help="Model name")
parser.add_argument("out_dir", type=str, default="exp/result",
                    help="Directory putting enhanced wav files")
parser.add_argument("--mix_dir", type=str, default=None,
                    help="Directory including mix wav files")
parser.add_argument("--mix_json", type=str, default=None,
                    help="Json file including mix wav files")
parser.add_argument('--device', default="cuda")
parser.add_argument("--sample_rate", default=8000,
                    type=int, help="Sample rate")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")


def save_wavs(estimate_source, mix_sig, lengths, filenames, out_dir, sr=16000):
    # Remove padding and flat
    flat_estimate = remove_pad(estimate_source, lengths)
    mix_sig = remove_pad(mix_sig, lengths)
    # Write result
    for i, filename in enumerate(filenames):
        filename = os.path.join(
            out_dir, os.path.basename(filename).strip(".wav"))
        write(mix_sig[i], filename + ".wav", sr=sr)
        C = flat_estimate[i].shape[0]
        # future support for wave playing
        for c in range(C):
            write(flat_estimate[i][c], filename + f"_s{c + 1}.wav")


def write(inputs, filename, sr=8000):
    librosa.output.write_wav(filename, inputs, sr, norm=True)


def get_mix_paths(args):
    mix_dir = None
    mix_json = None
    # fix mix dir
    try:
        if args.dset.mix_dir:
           mix_dir = args.dset.mix_dir
    except:
        mix_dir = args.mix_dir

    # fix mix json
    try:
        if args.dset.mix_json:
            mix_json = args.dset.mix_json
    except:
        mix_json = args.mix_json
    return mix_dir, mix_json


def separate(args, model=None, local_out_dir=None):
    mix_dir, mix_json = get_mix_paths(args)
    if not mix_json and not mix_dir:
        logger.error("Must provide mix_dir or mix_json! "
                     "When providing mix_dir, mix_json is ignored.")
    # Load model
    if not model:
        # model
        pkg = torch.load(args.model_path)
        if 'model' in pkg:
            model = pkg['model']
        else:
            model = pkg
        model = deserialize_model(model)
        logger.debug(model)
    model.eval()
    model.to(args.device)
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir

    # Load data
    eval_dataset = EvalDataset(
        mix_dir,
        mix_json,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
    )
    eval_loader = distrib.loader(
        eval_dataset, batch_size=1, klass=EvalDataLoader)

    if distrib.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    distrib.barrier()

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(eval_loader, ncols=120)):
            # Get batch data
            mixture, lengths, filenames = data
            mixture = mixture.to(args.device)
            lengths = lengths.to(args.device)
            # Forward
            estimate_sources = model(mixture)[-1]
            # save wav files
            save_wavs(estimate_sources, mixture, lengths,
                      filenames, out_dir, sr=args.sample_rate)


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    separate(args, local_out_dir=args.out_dir)
