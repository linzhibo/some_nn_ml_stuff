
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, image
import os
from sklearn.neural_network import MLPClassifier
import pickle
import time 
import math

epochs = 50
batch_size = 200

img_width, img_height = 28, 28


## generate data

def genData(prefix_name, number_folder):
    gen_number = 200
    train_datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.0,
        height_shift_range=0.1,
        rescale=1./255.,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

    for typ in ['train','validation']:
        number_path = typ + '/'+ number_folder
        path = '../data/digits/'+ number_path
        print ('generating data ... :' + typ + ' '+ number_folder)
        if not os.path.exists('./' + number_path):
            os.makedirs('./' + number_path)
        for filename in os.listdir(path):
            img = load_img(os.path.join(path, filename))
            x = img_to_array(img)  
            x = x.reshape((1,) + x.shape)
            count = 0

            for batch in train_datagen.flow(x, batch_size=1, save_prefix=prefix_name,save_to_dir=number_path, save_format='png'):
                count += 1
                if count > gen_number:
                    break 


####################################################################################

# # Creating CNN model

def finalModel():
    train_data_dir = 'train'
    validation_data_dir = 'validation'
    train_data = sorted(os.listdir(train_data_dir))
    x_train = []
    y_train = np.array([])

    for folder in train_data:
        folder_path = os.path.join(train_data_dir, folder)
        print ('loading train data from folder :'  + ' '+ folder)
        y_train = np.concatenate((y_train,np.array([int(folder)]*len(os.listdir(folder_path)))))
        for sample in os.listdir(folder_path):
            img_path = os.path.join(folder_path, sample)
            # print img_path
            x = image.load_img(img_path,color_mode='grayscale',target_size=(img_width,img_height))
            _, thresh = cv2.threshold(img_to_array(x),127,255,cv2.THRESH_BINARY )
            thresh = thresh.reshape(28,28,1)
            x_train.append(thresh)

    test_data = sorted(os.listdir(validation_data_dir))
    x_test = []
    y_test = np.array([])

    for folder in test_data:
        folder_path = os.path.join(validation_data_dir, folder)
        print ('loading test data from folder :'  + ' '+ folder)
        y_test = np.concatenate((y_test,np.array([int(folder)]*len(os.listdir(folder_path)))))
        for sample in os.listdir(folder_path):
            img_path = os.path.join(folder_path, sample)
            x = image.load_img(img_path,color_mode='grayscale', target_size=(img_width,img_height))
            _, thresh = cv2.threshold(img_to_array(x),127,255,cv2.THRESH_BINARY )
            thresh = thresh.reshape(28,28,1)
            x_test.append(thresh)

    x_train = np.array(x_train)
    x_test = np.array(x_test)   

    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    input_shape = (img_width,img_height,1)
    number_of_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train,  y_train,validation_split=0.01,epochs=epochs, shuffle=True,
                        batch_size = batch_size,validation_data= (x_test, y_test))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model.save('digit_classifier.h5')

#### hey i'm a separatooooooooooooooooooooooooooooooooooor ###


data = {'zero':'0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9'}
for x, y in data.items():
    genData(x,y)

finalModel()