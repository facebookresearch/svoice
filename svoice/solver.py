# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf

import json
import logging
from pathlib import Path
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from . import distrib
from .separate import separate
from .evaluate import evaluate
from .models.sisnr_loss import cal_loss
from .models.swave import SWave
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress


logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer
        if args.lr_sched == 'step':
            self.sched = StepLR(
                self.optimizer, step_size=args.step.step_size, gamma=args.step.gamma)
        elif args.lr_sched == 'plateau':
            self.sched = ReduceLROnPlateau(
                self.optimizer, factor=args.plateau.factor, patience=args.plateau.patience)
        else:
            self.sched = None

        # Training config
        self.device = args.device
        self.epochs = args.epochs
        self.max_norm = args.max_norm

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = Path(
            args.checkpoint_file) if args.checkpoint else None
        if self.checkpoint:
            logger.debug("Checkpoint will be saved to %s",
                         self.checkpoint.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        # keep track of losses
        self.history = []

        # Where to save samples
        self.samples_dir = args.samples_dir

        # logging
        self.num_prints = args.num_prints

        # for seperation tests
        self.args = args
        self._reset()

    def _serialize(self, path):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        torch.save(package, path)

    def _reset(self):
        load_from = None
        # Reset
        if self.checkpoint and self.checkpoint.exists() and not self.restart:
            load_from = self.checkpoint
        elif self.continue_from:
            load_from = self.continue_from

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_from == self.continue_from and self.args.continue_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])

            if 'optimizer' in package and not self.args.continue_best:
                self.optimizer.load_state_dict(package['optimizer'])
            self.history = package['history']
            self.best_state = package['best_state']

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch}: {info}")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(bold(f'Train Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid_loss = self._run_one_epoch(epoch, cross_valid=True)
            logger.info(bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))

            # learning rate scheduling
            if self.sched:
                if self.args.lr_sched == 'plateau':
                    self.sched.step(valid_loss)
                else:
                    self.sched.step()
                logger.info(
                    f'Learning rate adjusted: {self.optimizer.state_dict()["param_groups"][0]["lr"]:.5f}')

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss,
                       'valid': valid_loss, 'best': best_loss}
            # Save the best model
            if valid_loss == best_loss or self.args.keep_last:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())

            # evaluate and separate samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if (epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                with swap_state(self.model, self.best_state):
                    sisnr, pesq, stoi = evaluate(
                        self.args, self.model, self.tt_loader, self.args.sample_rate)
                metrics.update({'sisnr': sisnr, 'pesq': pesq, 'stoi': stoi})

                # separate some samples
                logger.info('Separate and save samples...')
                separate(self.args, self.model, self.samples_dir)

            self.history.append(metrics)
            info = " | ".join(
                f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize(self.checkpoint)
                    logger.debug("Checkpoint saved to %s",
                                 self.checkpoint.resolve())

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader,
                              updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            mixture, lengths, sources = [x.to(self.device) for x in data]
            estimate_source = self.dmodel(mixture)

            # only eval last layer
            if cross_valid:
                estimate_source = estimate_source[-1:]

            loss = 0
            cnt = len(estimate_source)
            # apply a loss function after each layer
            with torch.autograd.set_detect_anomaly(True):
                for c_idx, est_src in enumerate(estimate_source):
                    coeff = ((c_idx+1)*(1/cnt))
                    loss_i = 0
                    # SI-SNR loss
                    sisnr_loss, snr, est_src, reorder_est_src = cal_loss(
                        sources, est_src, lengths)
                    loss += (coeff * sisnr_loss)
                loss /= len(estimate_source)

                if not cross_valid:
                    # optimize model in training mode
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.max_norm)
                    self.optimizer.step()

            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))

            # Just in case, clear some memory
            del loss, estimate_source
        return distrib.average([total_loss / (i + 1)], i + 1)[0]
