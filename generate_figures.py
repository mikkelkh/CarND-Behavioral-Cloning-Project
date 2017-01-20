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
#if os.path.exists(dst_dir):
#    shutil.rmtree(dst_dir)
#os.makedirs(dst_dir)
#os.makedirs(os.path.join(dst_dir,"IMG"))

columns = ['center','left','right','angle','throttle','break','speed']

data_generated = pandas.DataFrame() # Container for the combined dataset

dirs = glob.glob(src_dir + "/*")
# Loop through all datasets
for idx,dir in enumerate(dirs):
    dataset = dir.split("/")[-1]
    
    if "left" in dir:    
    #if "straight_mikkel_1" in dir:    
    
        print(dir)
        csv_file = os.path.join(dir,"driving_log.csv")
        data = pandas.read_csv(csv_file,names=columns)
    
        data['center'] = data['center'].str.split("IMG/").str[1] # Remove first part of image path
        data['left'] = data['left'].str.split("IMG/").str[1] # Remove first part of image path
        data['right'] = data['right'].str.split("IMG/").str[1] # Remove first part of image path
    
        # Moving average filter on steering angles
        plt.figure(1,figsize=(6, 4))
        h_raw, = plt.plot(data['angle'][:200],color = '0.5')
        if moving_average != 0 and "udacity" not in dataset:
            print("Averaging angle data")
            win = (np.ones(moving_average)/moving_average)
            data['angle'] = signal.convolve(data['angle'],win,mode='same')
        h_avg, = plt.plot(data['angle'][:200],'k')
        plt.title('Moving average filter')
        plt.xlabel('sample id')
        plt.ylabel('angle')
        plt.legend([h_raw, h_avg], ['Raw', 'Averaged'])
    
        plt.figure(2,figsize=(6, 4))
        h_orig, = plt.plot(data['angle'][:200],'g')
        data[data['angle']>0] = float('nan')
        h_filtered, = plt.plot(data['angle'][:200],'r')
        plt.title('Filtering of recovery lap angles')
        plt.xlabel('sample id')
        plt.ylabel('angle')
        plt.legend([h_orig, h_filtered], ['Angles kept', 'Angles removed'])
    
        break






columns = ['center','left','right','angle','throttle','break','speed']
data = pandas.read_csv(os.path.join(dst_dir,"driving_log.csv"))
plt.figure(3,figsize=(5, 4))
plt.hist(data['angle'],21)
plt.title('Histogram of angles before horizontal flipping')
plt.xlabel('angle')
plt.ylabel('count')

data2 = data.copy()
data2['angle'] = -data2['angle']
data3 = pandas.concat([data,data2])
plt.figure(4,figsize=(5, 4))
plt.hist(data3['angle'],21)
plt.title('Histogram of angles after horizontal flipping')
plt.xlabel('angle')
plt.ylabel('count')
#print(data.shape[0])


print("Figure generation done")