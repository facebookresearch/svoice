import os
import glob
import random
import argparse
import itertools
import numpy as np
from pydub import AudioSegment

def create_mixture(s1_wav, s2_wav, noise_wav, out_dir, output_wav):
    # Load the audio files
    audio1 = AudioSegment.from_file(s1_wav)
    audio2 = AudioSegment.from_file(s2_wav)
    noise = AudioSegment.from_file(noise_wav)

    # Ensure all audio files have the same sample width and channels
    audio1 = audio1.set_sample_width(2)  # Modify the sample width if needed
    audio2 = audio2.set_sample_width(2)  # Modify the sample width if needed
    noise = noise.set_sample_width(2)    # Modify the sample width if needed

    audio1 = audio1.set_channels(1)  # Convert stereo to mono if needed
    audio2 = audio2.set_channels(1)  # Convert stereo to mono if needed
    noise = noise.set_channels(1)    # Convert stereo to mono if needed

    # Convert to numpy arrays
    data1 = np.array(audio1.get_array_of_samples())
    data2 = np.array(audio2.get_array_of_samples())
    noise_data = np.array(noise.get_array_of_samples())

    # Adjust the length of all audio signals to match the shortest one
    min_length = min(len(data1), len(data2), 500000)
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    # Repeat noise, if too small
    while(len(noise_data)<min_length):
        noise_data = np.concatenate((noise_data, np.array(noise.get_array_of_samples())))
    noise_data = noise_data[:min_length]

    # Mix the audio files and add noise
    mixed_data = data1 + data2 + noise_data  # Adjust the mixing process as needed

    # Create a new AudioSegment from the mixed data
    mixed_audio = AudioSegment(
        mixed_data.tobytes(),
        frame_rate=audio1.frame_rate,
        sample_width=audio1.sample_width,
        channels=audio1.channels
    )

    # Export the mixed audio with noise to a new file
    mixed_audio.export(f"{out_dir}/mix/{output_wav}", format="wav")
    audio1 = AudioSegment(data1.tobytes(),frame_rate=audio1.frame_rate,sample_width=audio1.sample_width,channels=audio1.channels)
    audio1.export(f"{out_dir}/s1/{output_wav}", format="wav")
    audio2 = AudioSegment(data2.tobytes(),frame_rate=audio1.frame_rate,sample_width=audio1.sample_width,channels=audio1.channels)
    audio2.export(f"{out_dir}/s2/{output_wav}", format="wav")

def create_dataset(num_samples, s1_folder, s2_folder, noise_folder, output_folder):
    s1_files = glob.glob(os.path.join(s1_folder, "*.wav"))
    s2_files = glob.glob(os.path.join(s2_folder, "*.wav"))
    noises = glob.glob(os.path.join(noise_folder, "*.wav"))
    mix2_pairs = list(itertools.product(s1_files, s2_files))
    random.shuffle(mix2_pairs)
    for s1_wav, s2_wav in mix2_pairs[:num_samples]:
        print(s1_wav, s2_wav)
        create_mixture(s1_wav,s2_wav,
                    random.choice(noises),
                    output_folder, 
                    f"{s1_wav.split('/')[-1]}_{s2_wav.split('/')[-1]}")

if __name__ == "__main__":
    #create_mixture("dataset/new_dataset/tr/s1/LJ001-0001.wav",
    #               "dataset/new_dataset/tr/s2/LJ002-0001.wav",
    #               "dataset/wham_noise/wham_noise/cv/01aa010k_1.3053_01po0310_-1.3053.wav",
    #               "mixed_audio_with_noise.wav")

    parser = argparse.ArgumentParser("Create mixture dataset from single voice dataset")
    parser.add_argument("--mix_size", default=100, type=int, help="Size for mixture dataset")
    parser.add_argument("--out_dir", type=str, default="dataset/new_out", help="Directory for the mixture dataset")
    parser.add_argument("--s1_dir", type=str, default="dataset/new_dataset/tr/s1", help="Directory of single voice")
    parser.add_argument("--s2_dir", type=str, default="dataset/new_dataset/tr/s2", help="Directory of single voice")
    parser.add_argument("--noise_dir", type=str, default="dataset/wham_noise/wham_noise/cv", help="Directory of noise")
    args = parser.parse_args()
    print(args)

    os.makedirs(f"{args.out_dir}/s1", exist_ok=True)
    os.makedirs(f"{args.out_dir}/s2", exist_ok=True)
    os.makedirs(f"{args.out_dir}/mix", exist_ok=True)
    create_dataset(args.mix_size, args.s1_dir, args.s2_dir, args.noise_dir, args.out_dir)