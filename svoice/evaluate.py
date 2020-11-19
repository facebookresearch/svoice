# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf and Alexandre Defossez (adefossez)

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys

import numpy as np
from pesq import pesq
from pystoi import stoi
import torch

from .models.sisnr_loss import cal_loss
from .data.data import Validset
from . import distrib
from .utils import bold, deserialize_model, LogProgress


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    'Evaluate separation performance using MulCat blocks')
parser.add_argument('model_path',
                    help='Path to model file created by training')
parser.add_argument('data_dir',
                    help='directory including mix.json, s1.json, s2.json, ... files')
parser.add_argument('--device', default="cuda")
parser.add_argument('--sdr', type=int, default=0)
parser.add_argument('--sample_rate', default=16000,
                    type=int, help='Sample rate')
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")


def evaluate(args, model=None, data_loader=None, sr=None):
    total_sisnr = 0
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = 5

    # Load model
    if not model:
        pkg = torch.load(args.model_path)
        if 'model' in pkg:
            model = pkg['model']
        else:
            model = pkg
        model = deserialize_model(model)
        if 'best_state' in pkg:
            model.load_state_dict(pkg['best_state'])
    logger.debug(model)
    model.eval()
    model.to(args.device)
    # Load data
    if not data_loader:
        dataset = Validset(args.data_dir)
        data_loader = distrib.loader(
            dataset, batch_size=1, num_workers=args.num_workers)
        sr = args.sample_rate
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                mixture, lengths, sources = [x.to(args.device) for x in data]
                # Forward
                with torch.no_grad():
                    mixture /= mixture.max()
                    estimate = model(mixture)[-1]
                sisnr_loss, snr, estimate, reorder_estimate = cal_loss(
                    sources, estimate, lengths)
                reorder_estimate = reorder_estimate.cpu()
                sources = sources.cpu()
                mixture = mixture.cpu()

                pendings.append(
                    pool.submit(_run_metrics, sources, reorder_estimate, mixture, None,
                                sr=sr))
                total_cnt += sources.shape[0]

            for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
                sisnr_i, pesq_i, stoi_i = pending.result()
                total_sisnr += sisnr_i
                total_pesq += pesq_i
                total_stoi += stoi_i

    metrics = [total_sisnr, total_pesq, total_stoi]
    sisnr, pesq, stoi = distrib.average(
        [m/total_cnt for m in metrics], total_cnt)
    logger.info(
        bold(f'Test set performance: SISNRi={sisnr:.2f} PESQ={pesq}, STOI={stoi}.'))
    return sisnr, pesq, stoi


def _run_metrics(clean, estimate, mix, model, sr, pesq=False):
    if model is not None:
        torch.set_num_threads(1)
        # parallel evaluation here
        with torch.no_grad():
            estimate = model(estimate)[-1]
    estimate = estimate.numpy()
    clean = clean.numpy()
    mix = mix.numpy()
    sisnr = cal_SISNRi(clean, estimate, mix)
    if pesq:
        pesq_i = cal_PESQ(clean, estimate, sr=sr)
        stoi_i = cal_STOI(clean, estimate, sr=sr)
    else:
        pesq_i = 0
        stoi_i = 0
    return sisnr.mean(), pesq_i, stoi_i


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    B, T = ref_sig.shape
    ref_sig = ref_sig - np.mean(ref_sig, axis=1).reshape(B, 1)
    out_sig = out_sig - np.mean(out_sig, axis=1).reshape(B, 1)
    ref_energy = (np.sum(ref_sig ** 2, axis=1) + eps).reshape(B, 1)
    proj = (np.sum(ref_sig * out_sig, axis=1).reshape(B, 1)) * \
        ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2, axis=1) / (np.sum(noise ** 2, axis=1) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr.mean()


def cal_PESQ(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, C, T]
        out_sig: numpy.ndarray, [B, C, T]
    Returns 
        PESQ
    """
    B, C, T = ref_sig.shape
    ref_sig = ref_sig.reshape(B*C, T)
    out_sig = out_sig.reshape(B*C, T)
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'nb')
    return pesq_val / (B*C)


def cal_STOI(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, C, T]
        out_sig: numpy.ndarray, [B, C, T]
    Returns:
        STOI
    """
    B, C, T = ref_sig.shape
    ref_sig = ref_sig.reshape(B*C, T)
    out_sig = out_sig.reshape(B*C, T)
    try:
        stoi_val = 0
        for i in range(len(ref_sig)):
            stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
        return stoi_val / (B*C)
    except:
        return 0


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [B, C, T]
        src_est: numpy.ndarray, [B, C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    avg_SISNRi = 0.0
    B, C, T = src_ref.shape
    for c in range(C):
        sisnr = cal_SISNR(src_ref[:, c], src_est[:, c])
        sisnrb = cal_SISNR(src_ref[:, c], mix)
        avg_SISNRi += (sisnr - sisnrb)
    avg_SISNRi /= C
    return avg_SISNRi


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    sisnr, pesq, stoi = evaluate(args)
    json.dump({'sisnr': sisnr,
               'pesq': pesq, 'stoi': stoi}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
