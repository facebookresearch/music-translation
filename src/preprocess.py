# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import data
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Input directory')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output directory')
    parser.add_argument('--norm-db', required=False, action='store_true')

    args = parser.parse_args()
    print(args)
    dataset = data.EncodedFilesDataset(args.input)
    dataset.dump_to_folder(args.output, norm_db=args.norm_db)



if __name__ == '__main__':
    main()
