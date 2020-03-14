import os, sys, glob

files = glob.glob("/media/Data/Datasets/Custom-dataset/Inference_folder/*")

for path in files:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    try:
        print(f"PROCESSING FILE: {path}")
        os.system(f'python predict.py -c config.json -i {path}')
    except Exception as e:
        print(f"FAILED FOR FILE: {path} due to ERROR: {e}")