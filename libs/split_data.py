from numpy.random import choice
from tqdm import tqdm
import argparse
import shutil
import os

parser = argparse.ArgumentParser(description="Dataset length normalizer")
parser.add_argument("-s", "--data-source", type=str,
                    required=True, help='Enter the path to the datset')
parser.add_argument("-r", "--val-split-ratio", type=float, default=0.2,
                    help='Enter the fraction of val data (default: 0.2 or 20%)', required=False)
parser.add_argument("-c", "--minimum-count", type=int, default=1000,
                    help='Enter the Forced Minimum Count', required=False)

args = vars(parser.parse_args())

all_data_path = args['data_source']
dir_names = [label for label in os.listdir(
    all_data_path) if os.path.isdir(os.path.join(all_data_path, label))]

dir_file_count = list()
for dir_name in dir_names:
    dir_file_count.append(
        len(os.listdir(os.path.join(all_data_path, dir_name))))

min_count = max(min(dir_file_count), args['minimum_count'])

for dir_name in dir_names:
    try:
        filenames = choice(os.listdir(os.path.join(
            all_data_path, dir_name)), size=min_count, replace=False)
    except:
        filenames = choice(os.listdir(os.path.join(
            all_data_path, dir_name)), size=min_count, replace=True)

    train_filenames = filenames[int(args["val_split_ratio"]*len(filenames)):]
    val_filenames = filenames[:int(args["val_split_ratio"]*len(filenames))]

    if not os.path.exists(os.path.join(all_data_path, 'train', dir_name)):
        os.makedirs(os.path.join(all_data_path, 'train', dir_name))

    if not os.path.exists(os.path.join(all_data_path, 'val', dir_name)) and args["val_split_ratio"] > 0.0:
        os.makedirs(os.path.join(all_data_path, 'val', dir_name))

    for filename in train_filenames:
        shutil.copy(os.path.join(all_data_path, dir_name, filename),
                    os.path.join(all_data_path, 'train', dir_name, filename))

    for filename in tqdm(val_filenames):
        shutil.copy(os.path.join(all_data_path, dir_name, filename),
                    os.path.join(all_data_path, 'val', dir_name, filename))

    print("Output Path: {}".format(os.path.join(all_data_path, 'train', dir_name)))
