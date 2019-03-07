import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
import pickle
import time 
import math
import sys
from sklearn.model_selection import GridSearchCV

img_width, img_height = 28, 28

def getTrainData():
    train_data_dir = 'train'
    train_data = sorted(os.listdir(train_data_dir))
    x_train = []
    y_train = np.array([])

    for folder in train_data:
        print ('loading data from folder :'  + ' '+ folder)
        folder_path = os.path.join(train_data_dir, folder)
        y_train = np.concatenate((y_train,np.array([int(folder)]*len(os.listdir(folder_path)))))
        for sample in os.listdir(folder_path):
            img_path = os.path.join(folder_path, sample)
            x = cv2.imread(img_path,0)     
            x = cv2.resize(x,(img_width,img_height))
            _, thresh = cv2.threshold(x,127,255,cv2.THRESH_BINARY )
            thresh = thresh.reshape(img_width*img_height,)
            x_train.append(thresh) 

    x_train = np.array(x_train)/255.
    return x_train, y_train

def getValidationData():
    validation_data_dir = 'validation'
    test_data = sorted(os.listdir(validation_data_dir))
    x_test = []
    y_test = np.array([])

    for folder in test_data:
        print ('loading data from folder :'  + ' '+ folder)
        folder_path = os.path.join(validation_data_dir, folder)
        y_test = np.concatenate((y_test,np.array([int(folder)]*len(os.listdir(folder_path)))))
        for sample in os.listdir(folder_path):
            img_path = os.path.join(folder_path, sample)
            x = cv2.imread(img_path,0)           
            x = cv2.resize(x,(img_width,img_height))
            _, thresh = cv2.threshold(x,127,255,cv2.THRESH_BINARY )
            thresh = thresh.reshape(img_width*img_height,)
            x_test.append(thresh) 

    x_test = np.array(x_test)/255.

    return x_test, y_test


def hyperParameterTuning():
    x_train, y_train = getTrainData()
    mlp = MLPClassifier(max_iter=100)
    parameter_space = {
    'hidden_layer_sizes': [(100,100),(7,),(10,10)],
    'activation': ['relu'],
    'solver': ['lbfgs','sgd', 'adam'],
    'alpha': [0.0001],
    'learning_rate': ['constant','adaptive'],
    'verbose':[True]}

    print ('begin training ...')
    
    clf = GridSearchCV(mlp, parameter_space, n_jobs=1, cv=3)
    clf.fit(x_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

def trainModel():
    x_train, y_train = getTrainData()
    print ('begin training ...')

    clf = mlp = MLPClassifier(hidden_layer_sizes=(100,100), 
                              activation='relu',
                              max_iter=200, 
                              alpha=0.0001,
                              solver='adam', 
                              verbose=10, 
                              tol=1e-4, 
                            #   random_state=1,
                              learning_rate='constant')
    clf.fit(x_train, y_train)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

def validationTest():
    load_model = pickle.load(open('finalized_model.sav', 'rb'))
    x_validation, y_validation = getValidationData()
    result = load_model.score(x_validation, y_validation)
    print result


def put_label(t_img,label,x,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 10
    l_y = int(y) + 10
    cv2.putText(t_img,str(label),(l_x,l_y), font,1.5,(255,0,0),2,cv2.LINE_AA)
    return t_img

def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows, cols = gray.shape 

    if rows > cols:
        factor = float(org_size)/float(rows)
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = float(org_size)/float(cols)
        cols = org_size
        rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    #apply apdding 
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    return gray

def getBoxedImg(img, load_model):
    l = len(img.shape)
    if (l < 3 ):
        (width,height) = img.shape
    elif (l == 3):
        (width,height,_) = img.shape
    img = cv2.resize(img, (180,270)) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 500 and cv2.contourArea(c) < 10000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h, x:x+w]
            roi = image_refiner(~roi)
            roi = roi.reshape(1,784)
            pred =  int(load_model.predict(roi)[0])
            (x,y),radius = cv2.minEnclosingCircle(c)
            img = put_label(img,pred,x,y)
    return img

# @profile
def imgTest():   
    load_model = pickle.load(open('finalized_model.sav', 'rb'))

    img = cv2.imread('../test_pics/1.png')
    crop = img[578:635,395:418]

    src = getBoxedImg(crop, load_model)  
    img[100:src.shape[0]+100,0:src.shape[1]] = src
    cv2.imshow('frame',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


# hyperParameterTuning()
trainModel()
validationTest()
imgTest()