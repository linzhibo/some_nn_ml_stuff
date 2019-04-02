import numpy as np
import caffe
import cv2 
import sys
import time 
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import thread
from obot_alpha_common.caffe_prediction import predict_material

labels = ['brick', 'concrete', 'asphalt', 'cobblestone', 'mud', 'flagstone', 'Undetermined']

video_path = '/home/zlin/Downloads/4 Seasons 2018 Cycling Video for Indoor Fat Burning Bike Training 60 Minute Ultra HD.mp4'

unit_prob = 1
prob = 0
mat_index = 0
material_prob = [unit_prob,unit_prob,unit_prob,unit_prob,unit_prob,unit_prob]




def gp(var1, var2):
    return 1./((1./var1) + (1./var2))

def filter(material_prob, mat_index, prob):
    for i in range(len(material_prob)):
        var1 = material_prob[i]
        if i == mat_index:
            var2 = max((1-prob) *100,5)
            material_prob[i] = gp(var1, var2)
        else:
            var2 = var1 +0.01
            material_prob[i] = var2
    return material_prob

def plotGaussian(material_prob, title=' '):
    x_axis = np.arange(-1, 6, 0.001)
    
    axes = plt.gca()
    axes.set_xlim([-1,6])
    axes.set_ylim([0,1.5])
    while(1):
        plt.cla()
        title = 'measurement: ' + str(labels[mat_index]) +'\n'+ ' probability: ' + str(prob)
        for i in range(len(material_prob)):
            plt.plot(x_axis, norm.pdf(x_axis,i,material_prob[i]), label = str(labels[i]))
        plt.title(title)
        plt.legend()
        plt.draw()
        plt.pause(0.03)


def videoFromFile(video_path):
    global material_prob, prob, mat_index
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    count = 0
    roi_size = 80
    vertical_displacement = 0
    horizontal_displacement = 0
    px1 = 720-roi_size - vertical_displacement
    px2 = 720 - vertical_displacement
    material = 'Undetermined'
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            road_roi = img[px1:px2 , (1280-roi_size)/2:(1280+roi_size)/2]
            mat_index, prob = predict_material(road_roi)
            material = labels[mat_index]
            material_prob = filter(material_prob, mat_index, prob)
            index = np.argmin(np.array(material_prob))

            cv2.rectangle(img, ((1280-roi_size)/2,px1), ((1280+roi_size)/2,px2),(0,255,0),2)
            cv2.putText(img,material,((1280+roi_size)/2, px2-roi_size), font,1.5,(255,0,0),1,cv2.LINE_AA)
            cv2.putText(img,str(prob),((1280+roi_size)/2, px2-roi_size+30), font,1,(255,0,0),1,cv2.LINE_AA)
            cv2.putText(img,labels[index],((1280+roi_size)/2, px2-roi_size+70), font,1.5,(0,255,0),1,cv2.LINE_AA)
            cv2.imshow("img", img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                sys.exit(1)
    cap.release()

try:
   thread.start_new_thread( videoFromFile, (video_path,) )
   thread.start_new_thread( plotGaussian, (material_prob, ))
except:
   print "Error: unable to start thread"
 
while 1:
   pass