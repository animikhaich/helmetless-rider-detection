import os, glob, shutil
from tqdm import tqdm
from random import choice

# Tunable Params
SOURCE_FOLDER = '/media/Data/Documents/Python-Codes/helmetless-rider-detection/data/captured_frames/tracker-multiclass/License Plate'
DEST_FOLDER = '/media/Data/Documents/Python-Codes/helmetless-rider-detection/data/selected_plates'


folders = os.listdir(SOURCE_FOLDER)

for folder in tqdm(folders):
    files = glob.glob(os.path.join(SOURCE_FOLDER, folder, "*"))
    source_path = choice(files)
    filename = os.path.split(source_path)[-1]
    dest_path = os.path.join(DEST_FOLDER, filename)
    shutil.move(source_path, dest_path)
    

