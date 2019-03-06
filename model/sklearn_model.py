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

def hyperParameterTuning():
    train_data_dir = 'train'
    validation_data_dir = 'validation'
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
            # cv2.imshow('frame',x)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     sys.exit(1)
            # x = image.load_img(img_path,color_mode='grayscale',target_size=(img_width,img_height))
            
            x = cv2.resize(x,(img_width,img_height))
            _, thresh = cv2.threshold(x,127,255,cv2.THRESH_BINARY )
            thresh = thresh.reshape(img_width*img_height,)
            x_train.append(thresh) 

    x_train = np.array(x_train)/255.
    
    mlp = MLPClassifier(max_iter=100)
    parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}

    print ('begin training ...')
    
    clf = GridSearchCV(mlp, parameter_space, n_jobs=1, cv=3)
    clf.fit(x_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


def skModel():
    train_data_dir = 'train'
    validation_data_dir = 'validation'
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
            # cv2.imshow('frame',x)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     sys.exit(1)
            # x = image.load_img(img_path,color_mode='grayscale',target_size=(img_width,img_height))
            
            x = cv2.resize(x,(img_width,img_height))
            _, thresh = cv2.threshold(x,127,255,cv2.THRESH_BINARY )
            thresh = thresh.reshape(img_width*img_height,)
            x_train.append(thresh) 

    x_train = np.array(x_train)/255.
    print ('begin training ...')

    clf = mlp = MLPClassifier(hidden_layer_sizes=(100,100), 
                              activation='relu',
                              max_iter=200, 
                              alpha=1e-4,
                              solver='sgd', 
                              verbose=10, 
                              tol=1e-4, 
                              random_state=1,
                              learning_rate_init=.1)
    clf.fit(x_train, y_train)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

def put_label(t_img,label,x,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 10
    l_y = int(y) + 10
    # cv2.rectangle(t_img,(l_x,l_y+5),(l_x+35,l_y-35),(0,255,0),-1) 
    cv2.putText(t_img,str(label),(l_x,l_y), font,1.5,(255,0,0),1,cv2.LINE_AA)
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
    # return cv2.resize(img.copy(), (height,width))
    return img

# @profile
def fitSKModel():   
    load_model = pickle.load(open('v1.sav', 'rb'))
    #img 9
    img = cv2.imread('/home/zlin/Desktop/Untitled Folder/7.png')
    crop = img[578:635,363:387]

    # img 6
    # img = cv2.imread('/home/zlin/Desktop/Untitled Folder/6.png')
    # crop = img[574:632,370:392]

    # #img 5
    # img = cv2.imread('/home/zlin/Desktop/Untitled Folder/5.png')
    # crop = img[578:635,377:401]

    #img 4
    # img = cv2.imread('/home/zlin/Desktop/Untitled Folder/4.png')
    # crop = img[578:635,384:407]

    #img 3
    # img = cv2.imread('/home/zlin/Desktop/Untitled Folder/2.png')
    # crop = img[578:635,389:413]

    #img 1
    # img = cv2.imread('/home/zlin/Desktop/Untitled Folder/1.png')
    # crop = img[578:635,395:418]

    src = getBoxedImg(crop, load_model)  
    img[100:src.shape[0]+100,0:src.shape[1]] = src
    cv2.imshow('frame',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# fitSKModel()

# skModel()
hyperParameterTuning()