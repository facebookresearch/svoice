# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Yossi Adi (adiyoss)

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
from .evaluate import _run_metrics


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    'Evaluate model automatic selection performance')
parser.add_argument('model_path_2spk',
                    help='Path to 2spk model file created by training')
parser.add_argument('model_path_3spk',
                    help='Path to 3spk model file created by training')
parser.add_argument('model_path_4spk',
                    help='Path to 4spk model file created by training')
parser.add_argument('model_path_5spk',
                    help='Path to 5spk model file created by training')
parser.add_argument(
    'data_dir', help='directory including mix.json, s1.json and s2.json files')
parser.add_argument('--device', default="cuda")
parser.add_argument('--sample_rate', default=8000,
                    type=int, help='Sample rate')
parser.add_argument('--thresh', default=0.001,
                    type=float, help='Threshold for model auto selection')
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")



# test pariwise matching
def pair_wise(padded_source, estimate_source):
    pair_wise = torch.sum(padded_source.unsqueeze(
        1)*estimate_source.unsqueeze(2), dim=3)
    if estimate_source.shape[1] != padded_source.shape[1]:
        idxs = pair_wise.argmax(dim=1)
        new_src = torch.FloatTensor(padded_source.shape)
        for b, idx in enumerate(idxs):
            new_src[b:, :, ] = estimate_source[b][idx]
        padded_source_pad = padded_source
        estimate_source_pad = new_src.cuda()
    else:
        padded_source_pad = padded_source
        estimate_source_pad = estimate_source
    return estimate_source_pad


def evaluate_auto_select(args):
    total_sisnr = 0
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = 5

    models = list()
    paths = [args.model_path_2spk, args.model_path_3spk,
             args.model_path_4spk, args.model_path_5spk]

    for path in paths:
        # Load model
        pkg = torch.load(path)
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
        models.append(model)

    # Load data
    dataset = Validset(args.data_dir)
    data_loader = distrib.loader(
        dataset, batch_size=1, num_workers=args.num_workers)
    sr = args.sample_rate
    y_hat = torch.zeros((4))

    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                mixture, lengths, sources = [x.to(args.device) for x in data]
                estimated_sources = list()
                reorder_estimated_sources = list()

                for model in models:
                    # Forward
                    with torch.no_grad():
                        raw_estimate = model(mixture)[-1]

                    estimate = pair_wise(sources, raw_estimate)
                    sisnr_loss, snr, estimate, reorder_estimate = cal_loss(
                        sources, estimate, lengths)
                    estimated_sources.insert(0, raw_estimate)
                    reorder_estimated_sources.insert(0, reorder_estimate)

                # =================== DETECT NUM. NON-ACTIVE CHANNELS ============== #
                selected_idx = 0
                thresh = args.thresh
                max_spk = 5
                mix_spk = 2
                ground = (max_spk - mix_spk)
                while (selected_idx <= ground):
                    no_sils = 0
                    vals = torch.mean(
                        (estimated_sources[selected_idx]/torch.abs(estimated_sources[selected_idx]).max())**2, axis=2)
                    new_selected_idx = max_spk - len(vals[vals > thresh])
                    if new_selected_idx == selected_idx:
                        break
                    else:
                        selected_idx = new_selected_idx
                if selected_idx < 0:
                    selected_idx = 0
                elif selected_idx > ground:
                    selected_idx = ground

                y_hat[ground - selected_idx] += 1
                reorder_estimate = reorder_estimated_sources[selected_idx].cpu(
                )
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
    logger.info(bold(f'Test set performance: SISNRi={sisnr:.2f} '
                     f'PESQ={pesq}, STOI={stoi}.'))
    logger.info(f'Two spks prob: {y_hat[0]/(total_cnt)}')
    logger.info(f'Three spks prob: {y_hat[1]/(total_cnt)}')
    logger.info(f'Four spks prob: {y_hat[2]/(total_cnt)}')
    logger.info(f'Five spks prob: {y_hat[3]/(total_cnt)}')
    return sisnr, pesq, stoi


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    sisnr, pesq, stoi = evaluate_auto_select(args)
    json.dump({'sisnr': sisnr,
               'pesq': pesq, 'stoi': stoi}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
