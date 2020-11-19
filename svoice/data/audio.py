# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Alexandre Défossez @adefossez, 2020

import json
from pathlib import Path
import math
import os
import tqdm
import sys

import torchaudio
import soundfile as sf
import torch as th
from torch.nn import functional as F


# If used, this should be saved somewhere as it takes quite a bit
# of time to generate
def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(os.path.abspath(file)))
    meta = []
    if progress:
        audio_files = tqdm.tqdm(audio_files,  ncols=80)
    for file in audio_files:
        siginfo, _ = torchaudio.info(file)
        length = siginfo.length // siginfo.channels
        meta.append((file, length))
    meta.sort()
    return meta


class Audioset:
    def __init__(self, files, length=None, stride=None, pad=True, augment=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.augment = augment
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(
                    math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            #  out = th.Tensor(sf.read(str(file), start=offset, frames=num_frames)[0]).unsqueeze(0)
            out = torchaudio.load(str(file), offset=offset,
                                  num_frames=num_frames)[0]
            if self.augment:
                out = self.augment(out.squeeze(0).numpy()).unsqueeze(0)
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            return out[0]


if __name__ == "__main__":
    json.dump(find_audio_files(sys.argv[1]), sys.stdout, indent=4)
    print()
