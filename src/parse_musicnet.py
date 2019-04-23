# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import csv
import os
from pathlib import Path

import pandas
from intervaltree import IntervalTree
from shutil import copy


def process_labels(root, path):
    trees = dict()
    for item in os.listdir(os.path.join(root,path)):
        if not item.endswith('.csv'): continue
        uid = int(item[:-4])
        tree = IntervalTree()
        with open(item, 'rb') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = int(label['start_time'])
                end_time = int(label['end_time'])
                instrument = int(label['instrument'])
                note = int(label['note'])
                start_beat = float(label['start_beat'])
                end_beat = float(label['end_beat'])
                note_value = label['note_value']
                tree[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)
        trees[uid] = tree
    return trees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='MusicNet directory')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output directory')

    args = parser.parse_args()
    print(args)

    src = args.input
    dst = args.output
    dst.mkdir(exist_ok=True, parents=True)

    domains = [
        ['Accompanied Violin', 'Beethoven'],
        ['Solo Cello', 'Bach'],
        ['Solo Piano', 'Bach'],
        ['Solo Piano', 'Beethoven'],
        ['String Quartet', 'Beethoven'],
        ['Wind Quintet', 'Cambini'],
    ]

    db = pandas.read_csv(src / 'musicnet_metadata.csv')
    traindir = src / 'musicnet/train_data'
    testdir = src / 'musicnet/test_data'

    for (ensemble, composer) in domains:
        fid_list = db[(db["composer"] == composer) & (db["ensemble"] == ensemble)].id.tolist()
        total_time = sum(db[(db["composer"] == composer) & (db["ensemble"] == ensemble)].seconds.tolist())
        print(f"Total time for {composer} with {ensemble} is: {total_time} seconds")

        domaindir = dst / f"{composer}_{ensemble.replace(' ', '_')}"
        if not os.path.exists(domaindir):
            os.mkdir(domaindir)

        for fid in fid_list:
            fname = traindir / f'{fid}.wav'
            if not fname.exists():
                fname = testdir / f'{fid}.wav'

            copy(str(fname), str(domaindir))


if __name__ == '__main__':
    main()