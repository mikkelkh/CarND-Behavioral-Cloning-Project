import glob
import pandas
import os
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from scipy import signal
#import cv2
from PIL import Image
from PIL import ImageOps
import shutil

resize = 0.5 # Resize image by this amount along x and y
moving_average = 5 # Angle average filter length

# Load data from src_dir containing potentially multiple data folders
src_dir = "../data/CarND-Behavioral-Cloning-Project/data"
# Generate one data folder combining data from all source folders
dst_dir = "../data/CarND-Behavioral-Cloning-Project/data_generated"

# Remove all files from folder before starting generation
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir)
os.makedirs(os.path.join(dst_dir,"IMG"))

columns = ['center','left','right','angle','throttle','break','speed']

data_generated = pandas.DataFrame() # Container for the combined dataset

dirs = glob.glob(src_dir + "/*")
# Loop through all datasets
for idx,dir in enumerate(dirs):
    dataset = dir.split("/")[-1]
    print(dir)
    csv_file = os.path.join(dir,"driving_log.csv")
    data = pandas.read_csv(csv_file,names=columns)

    data['center'] = data['center'].str.split("IMG/").str[1] # Remove first part of image path
    data['left'] = data['left'].str.split("IMG/").str[1] # Remove first part of image path
    data['right'] = data['right'].str.split("IMG/").str[1] # Remove first part of image path

    # Moving average filter on steering angles
    if moving_average != 0 and "udacity" not in dataset:
        print("Averaging angle data")
        win = (np.ones(moving_average)/moving_average)
        data['angle'] = signal.convolve(data['angle'],win,mode='same')

    # Filter recovery data such that only steering angles pointing towards the center of the road are used
    if "right" in dir:
        data = data[data['angle']<0] # Remove all zero and positive angles
    if "left" in dir:
        data = data[data['angle']>0] # Remove all zero and negative angles
    
    data_out = data.copy()      
    
    for cam in ['center','left','right']:
        data_out[cam] = dataset+"_"+data_out[cam]

    # Loop through all samples and resize images
    for index,sample in data.iterrows():
        for cam in ['center','left','right']:
            img = Image.open(os.path.join(dir,"IMG",sample[cam]))
            img = img.resize([int(float(img.size[0]) * resize), int(float(img.size[1]) * resize)],Image.ANTIALIAS)
            img.save(os.path.join(dst_dir,"IMG",data_out[cam][index]))

    # Append this dataset to the combined dataset
    data_generated = data_generated.append(data_out,ignore_index=True)

# Save new csv file
data_generated.to_csv(os.path.join(dst_dir,"driving_log.csv"),index=False)    

print("Data generation done")