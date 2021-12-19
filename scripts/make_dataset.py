# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Yossi Adi (adiyoss) and Shlomi E. Chazan (shlomke)

import argparse
import os
import numpy as np
import glob
import nprirgen
import soundfile as sf
import logging 

from tqdm import tqdm
from scipy import signal

def wavwrite_quantize(samples):
    return np.int16(np.round((2 ** 15) * samples))


def quantize(samples):
    int_samples = wavwrite_quantize(samples)
    return np.float64(int_samples) / (2 ** 15)


def wavwrite(file, samples, sr):
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype='PCM_16')


class dataset():
    def __init__(self, path, noise_path, sr, sec):
        self.path = path
        self.noise_path = noise_path
        self.sr = sr
        self.sec = sec
        self.size_of_signals = sec * sr  # fixed len of the signal 4secXfs
        # names of all the speakers
        self.speakers = glob.glob(os.path.join(self.path, "*"))
        self.len = len(self.speakers)  # num of the all speakers
        self.noise_names = glob.glob(os.path.join(self.noise_path, "*.wav"))
        self.noise_len = len(self.noise_names)

    def fetch_signals(self, number_of_speakers):
        # read random spealers
        self.num_of_speakers = number_of_speakers
        # (num_speakers,singlas)
        signals = np.zeros((self.num_of_speakers, self.size_of_signals))
        sig_indx = np.random.randint(
            low=0, high=self.len, size=self.num_of_speakers)
        names = list()
        for i in range(self.num_of_speakers):
            speaker = self.speakers[sig_indx[i]]
            cand_names = glob.glob(os.path.join(speaker + "/**","*.wav"))
            select_name_indx = np.random.randint(
            low=0, high=len(cand_names), size=1)
            name = cand_names[select_name_indx[0]]
            names.append(os.path.basename(name)[:-14])
            s, fs = sf.read(name)
            if len(s.shape) > 1:
                s = s[:, 0]
            s = quantize(s)

            l = len(s)
            if l > self.size_of_signals:
                noise_scale = np.std(s[0:self.size_of_signals]) / 80
                temp_noise = np.random.randn(self.size_of_signals,) * noise_scale
                signals[i] = temp_noise
                signals[i, :] = signals[i, 0:l] + s[0:self.size_of_signals]
            else:
                noise_scale = np.std(s) / 80
                temp_noise = np.random.randn(self.size_of_signals,) * noise_scale
                signals[i] = temp_noise
                signals[i, 0:l] = signals[i, 0:l] + s
        return signals, names

    def position_valid(self, room_dims, speaker_pos):
        if (speaker_pos[0] > 0.5 and speaker_pos[0] < room_dims[0]-0.5 and
                speaker_pos[1] > 0.5 and speaker_pos[1] < room_dims[1]-0.5):
            return 1
        else:
            return 0

    def room_gen(self, room_dims, RT60):
        mic_x = room_dims[0] // 2 + np.random.uniform(low=-0.15, high=0.15)
        mic_y = room_dims[1] // 2 + np.random.uniform(low=-0.15, high=0.15)
        mic_pos = [round(mic_x, 2), round(mic_y, 2), 1.5]  # x,y,z in meters

        speakers_pos = np.zeros(
            (self.num_of_speakers, 3))  # (num_speakers,x,y)
        speakers_angles = np.zeros((self.num_of_speakers, ))
        angel_deg = np.linspace(0, 180, 36)  # [0,30,60,90,120,150,180]
        for i in range(self.num_of_speakers):
            invalid = 0
            while not invalid:
                angle_index = np.random.randint(low=0, high=len(
                    angel_deg)) 
                speakers_angles[i] = angel_deg[angle_index]

                angle = np.deg2rad(angel_deg[angle_index])
                speaker_distance = 1.5 + \
                    round(np.random.uniform(low=-0.2, high=0.2), 2)
                x = mic_x + speaker_distance * np.cos(angle)
                y = mic_y + speaker_distance * np.sin(angle)
                speakers_pos[i, 0] = round(x, 2)
                speakers_pos[i, 1] = round(y, 2)
                speakers_pos[i, 2] = 1.5
                if self.position_valid(room_dims, speakers_pos[i]):
                    invalid = 1
                else:
                    logging.error('mic not in the room')
                    invalid = 0

        self.speakers_pos = speakers_pos
        self.speakers_angles = speakers_angles

        c = 340         # Sound velocity (m/s)
        r = [mic_pos]   # Receiver position [x y z] (m)

        # Room dimensions [x y z] (m)
        L = [room_dims[0], room_dims[1], room_dims[2]]
        rt = RT60                        # Reverberation time (s)
        n = 2048                         # Number of samples
        mtype = 'omnidirectional'        # Type of microphone
        order = -1                       # Reflection order
        dim = 3                          # Room dimension
        orientation = 0                  # Microphone orientation (rad)
        hp_filter = True                 # Enable high-pass filter
        Systems = np.zeros((self.num_of_speakers, n))
        Systems_anechoic = np.zeros((self.num_of_speakers, n))
        for i in range(self.num_of_speakers):
            s = speakers_pos[i]
            h_temp, _, _ = nprirgen.np_generateRir(
                L, s, r, soundVelocity=c, fs=self.sr, reverbTime=rt, nSamples=n, micType=mtype, nOrder=order, nDim=dim, isHighPassFilter=hp_filter)
            h_temp_anechoic, _, _ = nprirgen.np_generateRir(
                L, s, r, soundVelocity=c, fs=self.sr, reverbTime=0, nSamples=n, micType=mtype, nOrder=order, nDim=dim, isHighPassFilter=hp_filter)
            Systems[i] = h_temp
            Systems_anechoic[i] = h_temp_anechoic

        return Systems, Systems_anechoic

    def fetch_noise(self):
        noise_idex = np.random.randint(low=0, high=self.noise_len, size=1)
        name = self.noise_names[noise_idex[0]]

        s, fs = sf.read(name)
        if len(s.shape) > 1:
            s = s[:, 0]
        s = signal.resample(s, s.shape[0] // 2)
        s = quantize(s)
        if len(s) < self.size_of_signals:
            temp_noise = np.random.randn(self.size_of_signals,) * 0.01
            temp_noise[0:len(s)] = temp_noise[0:len(s)] + s[0:len(s)]
            return temp_noise
        else:
            return s[0:self.size_of_signals]

    def gen_scene(self, scenario_num_of_speakers, scene_i, current_write_path):
        S, sig_idx = self.fetch_signals(scenario_num_of_speakers)
        print(sig_idx)
        scenario_RT60 = round(np.random.uniform(low=0.1, high=1.0), 2)
        x = round(np.random.uniform(low=4, high=7), 2)
        y = round(np.random.uniform(low=4, high=7), 2)
        secnario_room_dims = [x, y, 2.5]
        H, H_anechoic = self.room_gen(secnario_room_dims, scenario_RT60)

        Mixed = np.zeros_like(S[0])
        conv_signals = np.zeros_like(S)
        angles_name = '_'
        for spk in range(scenario_num_of_speakers):
            temp_sig = np.convolve(S[spk], H[spk], mode='full')
            conv_signals[spk, :] = temp_sig[0: self.sec * self.sr]
            Mixed = Mixed +conv_signals[spk, :]
            angles_name = angles_name + str(int(self.speakers_angles[spk]))+'_'
        angles_name = angles_name[:-1]

        noise = self.fetch_noise()
        snr = round(np.random.uniform(low=0, high=15), 2)
        noise_gain = np.sqrt(10 ** (-snr/10) * np.std(Mixed) ** 2 / np.std(noise) ** 2)
        noise = noise_gain * noise
        Mixed = Mixed + noise

        mix_file_name = os.path.join(current_write_path, 'mix')
        if not os.path.exists(mix_file_name):
            os.mkdir(mix_file_name)
        filename = '_'.join(map(str, sig_idx))
        name = os.path.join(mix_file_name, str(scene_i) + angles_name + '_RT60_' + str(
            round(scenario_RT60, 2)) +
            '_snr_' + str(snr) + 
            '_fileidx_' + filename + '.wav')
        name = os.path.join(mix_file_name, str(scene_i) + angles_name + '_RT60_' + str(
            round(scenario_RT60, 2)) + '_fileidx_' + filename + '.wav')
        Mixed = Mixed / 1.2 / np.max(np.abs(Mixed))
        print(name)
        sf.write(name, Mixed, self.sr)

        for spk in range(scenario_num_of_speakers):
            target_file_name = os.path.join(current_write_path, 's' + str(spk+1))
            if not os.path.exists(target_file_name):
                os.mkdir(target_file_name)

            name = os.path.join(target_file_name, str(scene_i) + angles_name + '_RT60_' + str(
                round(scenario_RT60, 2)) + 
                '_snr_' + str(snr) + 
                '_fileidx_' + filename + '.wav')
            name = os.path.join(target_file_name, str(scene_i) + angles_name + '_RT60_' + str(
                round(scenario_RT60, 2)) + '_fileidx_' + filename + '.wav')
            s = np.convolve(S[spk], H_anechoic[spk], mode='full')
            s = s[0: self.sec * self.sr]
            s = s / 1.2 / np.max(np.abs(s))
            print(name)
            sf.write(name, s, self.sr)

def main(args):
    Data = dataset(args.in_path, args.noise_path, args.sr, args.sec)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    for i in tqdm(np.arange(args.num_of_scenes)):
        Data.gen_scene(args.num_of_speakers, i, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Mode")
    parser.add_argument('--in_path', type=str, default='dataset/wsj0-2spk/wav8k/min/tr/s1', help='')
    parser.add_argument('--out_path', type=str, default='dataset/tmp', help='')
    parser.add_argument('--noise_path', type=str, default='dataset/wham_noise/tr', help='')
    parser.add_argument('--num_of_speakers', type=int, default=2, help='no of speakers.')
    parser.add_argument('--num_of_scenes', type=int, default=10, help='no of examples.')
    parser.add_argument('--sec', type=int, default=4, help='')
    parser.add_argument('--sr', type=int, default=8000, help='')
    args = parser.parse_args()

    logging.info(args)
    main(args)
