import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Add, Input, BatchNormalization
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
imgMultiply = 1
croppe = 15

data = sorted(glob("data\\*.png"),key=os.path.getmtime)

for image in tqdm(data):
    try:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    except:
        pass
# Prepare Data
    img  = cv2.resize(img,(80,64))
    img  = img[offset:, :]

    img  = img.astype("float32")
    img /= 255

    img = img.reshape((80,64-offset ,1))
    a = image.split("_")[0].split("\\")[1]
    direct  = int(a)-2

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

regularisationParam = 1e-3


inp   = Input(shape=(80, 64-croppe ,3,))

conv  = Conv2D(32, (2,2), strides=2, activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(inp)
conv  = BatchNormalization()(conv)
conv  = MaxPooling2D(pool_size=(2,2), strides=2)(conv)
conv  = Dropout(0.2)(conv)

conv  = Conv2D(64, (2,2), strides=2, activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(conv)
conv  = BatchNormalization()(conv)
conv  = MaxPooling2D(pool_size=(2,2), strides=2)(conv)
conv  = Dropout(0.2)(conv)

conv  = Flatten()(conv)

dens  = Dense(64, use_bias=False,  activation="relu", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(conv)
dens  = BatchNormalization()(dens)
dens  = Dense(16  , use_bias=False, activation="relu", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(dens)
dens  = BatchNormalization()(dens)
dens  = Dense(8  , use_bias=False, activation="relu", kernel_regularizer=keras.regularizers.l2(l=regularisationParam))(dens)    #relu
dens  = BatchNormalization()(dens)

output = Dense(1,  use_bias=False, activation="tanh")(dens)

model = Model(inputs=inp, outputs=output)

# Entrainer Model

# model.summary()

model.compile(loss="mean_squared_error",
	optimizer=Adam(lr=0.0001),
	metrics=["accuracy"])

dataGenerator = ImageDataGenerator(brightness_range=[0.5,1.5])
dataGenerator.fit(x_train)

iterator = dataGenerator.flow(x_train, y_train, batch_size=32)

history = model.fit_generator(iterator, steps_per_epoch=120,
    epochs = 200,
    validation_data = (x_test,y_test))

#history = model.fit(x_train,y_train,
#	batch_size = 16,
#	epochs = 200,
#	validation_data = (x_test,y_test))

model.save("IronCarHistory.h5")

def graphValues(history,Acc):
    plt.figure()
    if Acc == 1:
        plt.plot(history.history['accuracy'],color="green")
        plt.plot(history.history['val_accuracy'],color="blue")
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

    elif Acc == 0:
        plt.plot(history.history['loss'],color="orange")
        plt.plot(history.history['val_loss'],color="red")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

graphValues(history,0)
graphValues(history,1)
plt.show()
