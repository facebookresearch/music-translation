# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import data
import argparse
from pathlib import Path
import tqdm

from utils import inv_mu_law, save_audio
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, nargs='*',
                        help='Path to data dir')
    parser.add_argument('--data-from-args', type=Path,
                        help='Path to args.pth')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output path')
    parser.add_argument('-n', type=int,
                        help='Num samples to make')
    parser.add_argument('--seq-len', type=int, default=80000)

    args = parser.parse_args()

    if args.data:
        dataset_paths = args.data
    elif args.data_from_args:
        input_args, _ = torch.load(args.data_from_args)
        dataset_paths = input_args.data
    else:
        print('Please supply either --data or --data-from-args')
        return

    if dataset_paths[0].is_file():
        datasets = [data.H5Dataset(dataset_paths[0], args.seq_len, 'wav')]
    else:
        datasets = [data.H5Dataset(p / 'test', args.seq_len, 'wav')
                    for p in dataset_paths]

    for dataset_id, dataset in enumerate(datasets):
        for i in tqdm.trange(args.n):
            wav_data, _ = dataset[0]
            wav_data = inv_mu_law(wav_data.numpy())
            save_audio(wav_data, args.output / f'{dataset_id}/{i}.wav', rate=data.EncodedFilesDataset.WAV_FREQ)


if __name__ == '__main__':
    main()
