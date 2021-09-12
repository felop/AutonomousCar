import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Add, Input, BatchNormalization, Layer
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import h5py, sys, os
import cv2
from tqdm import tqdm
from glob import glob
from random import random
import polygon
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class randomize(Layer):
    def __init__(self, Brightrange, **kwargs):
#        self.output_dim = self.
        self.Brightrange = Brightrange
        super(randomize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(randomize, self).build(input_shape)

    def call(self, x):
        if K.learning_phase() == 1:
            #return x
            #return x*(random()*self.Brightrange[1]+self.Brightrange[0])
            #return polygon.shadow(x,2,2)
            return polygon.shadow(x*(random()*self.Brightrange[1]+self.Brightrange[0]), 4, 1, 0.95) # img ; random factor ; number of polygons ; alpha
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

x_train  = []
x_train1 = []
x_train2 = []
x_train3 = []
x_trainLast = []
y_train  = []
y_trainLast = []
x_test   = []
x_test1  = []
x_test2  = []
x_test3  = []
y_test   = []
directInv = 0

offset = 15

data = sorted(glob("whitePics\\*.png"),key=os.path.getmtime)

for image in tqdm(data):
        img = cv2.imread(image,cv2.IMREAD_COLOR)#, cv2.IMREAD_GRAYSCALE)
# Prepare Data
    img  = cv2.resize(img,(80,64))
    img  = img[offset:, :]

    img  = img.astype("float32")
    img /= 255

    img = img.reshape((64-offset,80 ,3))
    a = image.split("_")[0].split("\\")[1]
    direct  = int(a)

    x_train.append(img)
    y_train.append(direct)

#x_train,y_train = shuffle(x_train,y_train)
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train, test_size = 0.20)

#train
x_train1 = x_train.copy()
x_train2 = x_train.copy()
x_train3 = x_train.copy()
#test
x_test1 = x_test.copy()
x_test2 = x_test.copy()
x_test3 = x_test.copy()

#train
del x_train1[0],  x_train1[0]     #Image shift
del x_train2[0],  x_train2[-1]
del x_train3[-1], x_train3[-1]
del y_train[-1],  y_train[-1]     #label shift
#test
del x_test1[0],  x_test1[0]     #Image shift
del x_test2[0],  x_test2[-1]
del x_test3[-1], x_test3[-1]
del y_test[-1],  y_test[-1]     #label shift

#train
x_train1 = np.array(x_train1)
x_train2 = np.array(x_train2)
x_train3 = np.array(x_train3)
y_train = np.array(y_train)
#test
x_test1 = np.array(x_test1)
x_test2 = np.array(x_test2)
x_test3 = np.array(x_test3)
y_test = np.array(y_test)

# concatenate (merge the 3 one channel images into a 3 channel image)
x_train = np.concatenate((x_train1,x_train2,x_train3),3)
x_test = np.concatenate((x_test1,x_test2,x_test3),3)
# data augmentation

# Model

#conv  = Dropout(0.2)(conv)

regularisationParam = 1e-9

inp   = Input(shape=(64-offset,80 ,9,))

dataGen = randomize([0.8,1.2])(inp)

conv  = Conv2D(32, (2,2), strides=2, activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(dataGen)
conv  = BatchNormalization()(conv)
conv  = MaxPooling2D(pool_size=(2,2), strides=2)(conv)
conv  = Dropout(0.2)(conv)

conv  = Conv2D(64, (2,2), strides=2, activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(conv)
conv  = BatchNormalization()(conv)
conv  = MaxPooling2D(pool_size=(2,2), strides=2)(conv)
conv  = Dropout(0.2)(conv)

conv  = Flatten()(conv)

dens  = Dense(32, use_bias=False,  activation="relu", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(conv)
dens  = BatchNormalization()(dens)
dens  = Dense(16  , use_bias=False, activation="relu", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(dens)
dens  = BatchNormalization()(dens)

output = Dense(1,  use_bias=False, activation="tanh")(dens)

model = Model(inputs=inp, outputs=output)

# Entrainer Model

model.summary()
model.compile(loss="mean_squared_error",
	optimizer=Adam(lr=0.0001),
	metrics=["accuracy"])

history = model.fit(x_train,y_train,
	batch_size = 32,
	epochs = 80,
	validation_data = (x_test,y_test))

def graphValues(history,Acc):
    plt.figure()
    if Acc == 1:
        plt.plot(history.history['accuracy'],color="green")
        plt.plot(history.history['val_accuracy'],color="blue")
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.ylim([0,1])
        plt.legend(['train', 'test'], loc='upper left')

    elif Acc == 0:
        plt.plot(history.history['loss'],color="orange")
        plt.plot(history.history['val_loss'],color="red")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim([0,1])
        plt.legend(['train', 'test'], loc='upper left')

graphValues(history,0)
graphValues(history,1)
plt.show()

if input("wanna save (yes/no) ? ") == "yes":
    model.save("IronCarHistory.h5")
