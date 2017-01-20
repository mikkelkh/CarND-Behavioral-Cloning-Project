import glob
import os
import pandas
import tensorflow
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.layer_utils import layer_from_config
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#import cv2
from PIL import Image
import numpy as np
import json

batch_size = 64
nb_epoch = 20

validation_rate = 0.1
test_rate = 0.1

# Regularization
dropout_rate = 0.4

# Data augmentation
horizontal_flip = True # Use horizontal flipping of image (and corresponding adjustment of steering angle)
cams = ['left','center','right'] # Use all three cameras

# Open data and shuffle
data_dir = "../data/CarND-Behavioral-Cloning-Project/data_generated"
data = pandas.read_csv(os.path.join(data_dir,"driving_log.csv"))
data = shuffle(data)

# Split data into training, validation, and test
data_train = data[:int(data.shape[0]*(1-validation_rate-test_rate))]
data_train = data_train.reset_index(drop=True)
data_val = data[data_train.shape[0]:int(data.shape[0]*(1-test_rate))]
data_val = data_val.reset_index(drop=True)
data_test = data[data_train.shape[0]+data_val.shape[0]:]
data_test = data_val.reset_index(drop=True)

print("training size = {}".format(data_train.shape[0]))
print("validation size = {}".format(data_val.shape[0]))
print("test size = {}".format(data_test.shape[0]))

# Construct python generator that generates batches on-the-fly with data augmentation
def myGenerator(data,batch_size=32,horizontal_flip = True):
    # Select one of the three cameras (left,center,right) randomly and apply angle corrections for the left and right
    left_right_cam_adj = 0.3
    adjustments = {'left': left_right_cam_adj, 'center': 0, 'right': -left_right_cam_adj}

    # Get image size in order to allocate memory for X_train
    img_size  = list(np.asarray(Image.open(os.path.join(data_dir,"IMG",data['center'][0]))).shape)
    while 1:
        # Shuffle the training data for each epoch
        data = data.reindex(np.random.permutation(data.index))
        for i in range(0,data.shape[0],batch_size):
            if i+batch_size <= data.shape[0]: # Make sure all batches are same size (required by keras)
                X = np.empty([batch_size]+img_size, dtype=float)
                y = np.empty(batch_size)
                for index,(_,sample) in enumerate(data.iloc[i:i+batch_size].iterrows()):
                    cam = np.random.choice(cams,1)[0] # Select random camera (left,center,right)
                    angle = sample['angle'] + adjustments[cam] # Adjust angle accordingly
                    img = np.asarray(Image.open(os.path.join(data_dir,"IMG",sample[cam]))).astype(float) # Load image
                    # Flip image and angle horizontally every second time
                    if horizontal_flip and np.random.randint(2)==0:
                        img = np.fliplr(img)
                        angle = -angle

                    X[index, :] = img
                    y[index] = angle
                yield X,y

input_size = next(myGenerator(data_train))[0].shape[1:4]

# Define model architecture
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(input_size)))
model.add(Convolution2D(8, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2),border_mode='valid'))

model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

model.add(Convolution2D(128, 3, 3, border_mode='valid'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))

model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

# Train model and report validation loss for each epoch
history = model.fit_generator(myGenerator(data_train,batch_size), samples_per_epoch=data_train.shape[0], nb_epoch = nb_epoch, verbose=2, callbacks=[], validation_data=myGenerator(data_val,batch_size), nb_val_samples=data_val.shape[0])

epochs = np.asarray(history.epoch)+1
plt.figure(1,figsize=(8, 4))
train_loss, = plt.plot(epochs,history.history['loss'],color="b")
val_loss, = plt.plot(epochs,history.history['val_loss'],color="#ffa500")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend([train_loss, val_loss], ['Training loss', 'Validation loss'])

# Evaluate model on test data
test_loss = model.evaluate_generator(myGenerator(data_test,batch_size), val_samples=data_test.shape[0])
print("Test loss={}".format(test_loss))

json_string = model.to_json()
text_file = open("model.json", "w")
text_file.write(json_string)
text_file.close()
model.save_weights('model.h5')

print("Done")