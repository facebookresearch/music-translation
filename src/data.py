# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import librosa
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from scipy.io import wavfile

import utils
from utils import mu_law

logger = utils.setup_logger(__name__, 'data.log')


def random_of_length(seq, length):
    limit = seq.size(0) - length
    if length < 1:
        # logging.warning("%d %s" % (length, path))
        return None

    start = random.randint(0, limit)
    end = start + length
    return seq[start: end]


class EncodedFilesDataset(data.Dataset):
    """
    Uses ffmpeg to read a random short segment from the middle of an encoded file
    """
    FILE_TYPES = ['mp3', 'ape', 'm4a', 'flac', 'mkv', 'wav']
    WAV_FREQ = 16000
    INPUT_FREQ = 44100
    FFT_SZ = 2048
    WINLEN = FFT_SZ - 1
    HOP_SZ = 80

    def __init__(self, top, seq_len=None, file_type=None, epoch_len=10000):
        self.path = Path(top)
        self.seq_len = seq_len
        self.file_types = [file_type] if file_type else self.FILE_TYPES
        self.file_paths = self.filter_paths(self.path.glob('**/*'), self.file_types)
        self.epoch_len = epoch_len

    @staticmethod
    def filter_paths(haystack, file_types):
        return [f for f in haystack
                if (f.is_file()
                    and any(f.name.endswith(suffix) for suffix in file_types)
                    and '__MACOSX' not in f.parts)]

    def _random_file(self):
        # return np.random.choice(self.file_paths, p=self.probs)
        return random.choice(self.file_paths)

    @staticmethod
    def _file_length(file_path):
        output = subprocess.run(['ffprobe',
                                 '-show_entries', 'format=duration',
                                 '-v', 'quiet',
                                 '-print_format', 'compact=print_section=0:nokey=1:escape=csv',
                                 str(file_path)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).stdout
        duration = float(output)

        return duration

    def _file_slice(self, file_path, start_time):
        length_sec = self.seq_len / self.WAV_FREQ
        length_sec += .01  # just in case
        with NamedTemporaryFile() as output_file:
            output = subprocess.run(['ffmpeg',
                                     '-v', 'quiet',
                                     '-y',  # overwrite
                                     '-ss', str(start_time),
                                     '-i', str(file_path),
                                     '-t', str(length_sec),
                                     '-f', 'wav',
                                     # '-af', 'dynaudnorm',
                                     '-ar', str(self.WAV_FREQ),  # audio rate
                                     '-ac', '1',  # audio channels
                                     output_file.name
                                     ],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE).stdout
            rate, wav_data = wavfile.read(output_file)
            assert wav_data.dtype == np.int16
            wav = wav_data[:self.seq_len].astype('float')

            return wav

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, _):
        wav = self.random_file_slice()
        return torch.FloatTensor(wav)

    def random_file_slice(self):
        wav_data = None

        while wav_data is None or len(wav_data) != self.seq_len:
            try:
                file, file_length_sec, start_time, wav_data = self.try_random_file_slice()
            except Exception as e:
                logger.exception('Exception %s in random_file_slice.', e)

        # logger.debug('Sample: File: %s, File length: %s, Start time: %s',
        #              file, file_length_sec, start_time)

        return wav_data

    def try_random_file_slice(self):
        file = self._random_file()
        file_length_sec = self._file_length(file)
        segment_length_sec = self.seq_len / self.WAV_FREQ
        if file_length_sec < segment_length_sec:
            logger.warn('File "%s" has length %s, segment length is %s',
                        file, file_length_sec, segment_length_sec)

        start_time = random.random() * (file_length_sec - segment_length_sec * 2)  # just in case
        try:
            wav_data = self._file_slice(file, start_time)
        except Exception as e:
            logger.info(f'Exception in file slice: {e}. '
                        f'File: {file}, '
                        f'File length: {file_length_sec}, '
                        f'Start time: {start_time}')
            raise

        if len(wav_data) != self.seq_len:
            logger.warn('File "%s" has length %s, segment length is %s, wav data length: %s',
                        file, file_length_sec, segment_length_sec, len(wav_data))

        return file, file_length_sec, start_time, wav_data

    def dump_to_folder(self, output: Path, norm_db=False):
        for file_path in tqdm.tqdm(self.file_paths):
            output_file_path = output / file_path.relative_to(self.path).with_suffix('.h5')
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(suffix='.wav') as output_wav_file, \
                    NamedTemporaryFile(suffix='.wav') as norm_file_path, \
                    NamedTemporaryFile(suffix='.wav') as wav_convert_file:
                if norm_db:
                    logger.debug(f'Converting {file_path} to {wav_convert_file.name}')
                    subprocess.run(['ffmpeg',
                                    '-y',
                                    '-i', file_path,
                                    wav_convert_file.name],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

                    logger.debug(f'Companding {wav_convert_file.name} to {norm_file_path.name}')
                    subprocess.run(['sox',
                                    '-G',
                                    wav_convert_file.name,
                                    norm_file_path.name,
                                    'compand',
                                    '0.3,1',
                                    '6:-70,-60,-20',
                                    '-5',
                                    '-90',
                                    '0.2'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
                    input_file_path = norm_file_path.name
                else:
                    input_file_path = file_path

                logger.debug(f'Converting {input_file_path} to {output_wav_file.name}')
                subprocess.run(['ffmpeg',
                                '-v', 'quiet',
                                '-y',  # overwrite
                                '-i', input_file_path,
                                # '-af', 'dynaudnorm',
                                '-f', 'wav',
                                '-ar', str(self.WAV_FREQ),  # audio rate
                                '-ac', '1',  # audio channels,
                                output_wav_file.name
                                ],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
                try:
                    rate, wav_data = wavfile.read(output_wav_file.name)
                except ValueError:
                    logger.info(f'Cannot read {file_path} wav conversion')
                    raise
                    # raise
                assert wav_data.dtype == np.int16
                wav = wav_data.astype('float')

                with h5py.File(output_file_path, 'w') as output_file:
                    chunk_shape = (min(10000, len(wav)),)
                    wav_dset = output_file.create_dataset('wav', wav.shape, dtype=wav.dtype,
                                                          chunks=chunk_shape)
                    wav_dset[...] = wav

                logger.debug(f'Saved input {file_path} to {output_file_path}. '
                             f'Wav length: {wav.shape}')


class H5Dataset(data.Dataset):
    def __init__(self, top, seq_len, dataset_name, epoch_len=10000, augmentation=None, short=False,
                 whole_samples=False, cache=False):
        self.path = Path(top)
        self.seq_len = seq_len
        self.epoch_len = epoch_len
        self.short = short
        self.whole_samples = whole_samples
        self.augmentation = augmentation
        self.dataset_name = dataset_name

        self.file_paths = list(self.path.glob('**/*.h5'))
        if self.short:
            self.file_paths = [self.file_paths[0]]

        self.data_cache = {}
        if cache:
            for file_path in tqdm.tqdm(self.file_paths,
                                       desc=f'Reading dataset {top.parent.name}/{top.name}'):
                dataset = self.read_h5_file(file_path)
                self.data_cache[file_path] = dataset[:]

        if not self.file_paths:
            logger.error(f'No files found in {self.path}')

        logger.info(f'Dataset created. {len(self.file_paths)} files, '
                    f'augmentation: {self.augmentation is not None}. '
                    f'Path: {self.path}')

    def __getitem__(self, _):
        ret = None
        while ret is None:
            try:
                ret = self.try_random_slice()
                if self.augmentation:
                    ret = [ret, self.augmentation(ret)]
                else:
                    ret = [ret, ret]

                if self.dataset_name == 'wav':
                    ret = [mu_law(x / 2 ** 15) for x in ret]
            except Exception as e:
                logger.info('Exception %s in dataset __getitem__, path %s', e, self.path)
                logger.debug('Exception in H5Dataset', exc_info=True)

        return torch.tensor(ret[0]), torch.tensor(ret[1])

    def try_random_slice(self):
        h5file_path = random.choice(self.file_paths)
        if h5file_path in self.data_cache:
            dataset = self.data_cache[h5file_path]
        else:
            dataset = self.read_h5_file(h5file_path)
        return self.read_wav_data(dataset, h5file_path)

    def read_h5_file(self, h5file_path):
        try:
            f = h5py.File(h5file_path, 'r')
        except Exception as e:
            logger.exception('Failed opening %s', h5file_path)
            raise

        try:
            dataset = f[self.dataset_name]
        except Exception:
            logger.exception(f'No dataset named {self.dataset_name} in {file_path}. '
                             f'Available datasets are: {list(f.keys())}.')

        return dataset

    def read_wav_data(self, dataset, path):
        if self.whole_samples:
            data = dataset[:]
        else:
            length = dataset.shape[0]

            if length <= self.seq_len:
                logger.debug('Length of %s is %s', path, length)

            start_time = random.randint(0, length - self.seq_len)
            data = dataset[start_time: start_time + self.seq_len]
            assert data.shape[0] == self.seq_len

        return data.T

    def __len__(self):
        return self.epoch_len


class WavFrequencyAugmentation:
    def __init__(self, wav_freq, magnitude=0.5):
        self.magnitude = magnitude
        self.wav_freq = wav_freq

    def __call__(self, wav):
        length = wav.shape[0]
        perturb_length = random.randint(length // 4, length // 2)
        perturb_start = random.randint(0, length // 2)
        perturb_end = perturb_start + perturb_length
        pitch_perturb = (np.random.rand() - 0.5) * 2 * self.magnitude

        ret = np.concatenate([wav[:perturb_start],
                              librosa.effects.pitch_shift(wav[perturb_start:perturb_end],
                                                          self.wav_freq, pitch_perturb),
                              wav[perturb_end:]])

        return ret


class DatasetSet:
    def __init__(self, dir: Path, seq_len, args):
        if args.data_aug:
            augmentation = WavFrequencyAugmentation(EncodedFilesDataset.WAV_FREQ, args.magnitude)
        else:
            augmentation = None

        self.train_dataset = H5Dataset(dir / 'train', seq_len, epoch_len=10000000000,
                                       dataset_name=args.h5_dataset_name, augmentation=augmentation,
                                       short=args.short, cache=False)
        self.train_loader = data.DataLoader(self.train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

        self.train_iter = iter(self.train_loader)

        self.valid_dataset = H5Dataset(dir / 'val', seq_len, epoch_len=1000000000,
                                       dataset_name=args.h5_dataset_name, augmentation=augmentation,
                                       short=args.short)
        self.valid_loader = data.DataLoader(self.valid_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers // 10 + 1,
                                            pin_memory=True)

        self.valid_iter = iter(self.valid_loader)
