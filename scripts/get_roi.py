import math
import sys
import cv2
import os
import numpy as np
from ros_ld import *

img_folder = '/media/zlin/DATA/Dataset/data_road/testing/image_2'
video_path = '/home/zlin/Downloads/30 Minute Sunshine Beach Relax Cycling Training Spain 4K Video 2018.mp4'
M, Minv = create_M()
count = 0
keep_writing = 0 
material = 'mud'
materials = ['asphalt', 'brick', 'cobblestone', 'concrete', 'flagstone', 'mud']

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total: 
        print ""

def nothing(x):
    pass

def showImgWithRoi(img):
    '''
    trackbar to move roi vertically and horizontally
    third trackbar to choose material to save
    space to skip
    esc to quit
    y to save current roi
    k to keep/stop saving 
    '''
    global count, keep_writing
    import sys
    height, width, ch = img.shape
    cv2.namedWindow("img")
    cv2.moveWindow("img", 200,200)
    roi_size = 80
    cv2.createTrackbar('horizontal','img',0,width-roi_size,nothing)
    cv2.createTrackbar('vertical','img',0,height-roi_size,nothing)
    # switch = '0 : asphalt \n1: brick \n2: cobblestone \n3: oncrete \n4: flagstone \n5: mud'
    switch = 'material'
    cv2.createTrackbar(switch, 'img', 0,5,nothing)
    # cv2.setTrackbarPos('vertical', 'img',15) 
      

    while(1):
        src  = img.copy()
        vertical = cv2.getTrackbarPos('vertical', 'img')
        horizontal = cv2.getTrackbarPos('horizontal', 'img')
        material = materials[cv2.getTrackbarPos(switch, 'img')]
        road_roi = img[vertical: roi_size+vertical, horizontal: roi_size+horizontal]
        cv2.rectangle(src, (horizontal,vertical),(roi_size+horizontal,roi_size+vertical), (0,255,0),2)
        cv2.imshow("img", src)
        if keep_writing:
            k = cv2.waitKey(1) & 0xFF
            if k == 107:
                keep_writing = 0
            cv2.imwrite(material+'/'+str(count) + '.png', road_roi)
            print "saved "+ str(count) + '.png' + " ..."
            count += 1
            return
        k = cv2.waitKey(1) & 0xFF
        if k == 107:
            keep_writing = 1
        if k == 121:
            cv2.imwrite(material+'/'+str(count) + '.png', road_roi)
            print "saved "+ str(count) + '.png' + " ..."
            count += 1
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(1)
        if k == 32:
            return
        
        


def extract_img(img_folder):
    for filename in os.listdir(img_folder):
        img = cv2.imread(os.path.join(img_folder, filename))
        img = cv2.resize(img, (1280,720))
        warp = transform(img, M)
        roi_size = 80
        road_roi = warp[ 720-roi_size: 720, (1280-roi_size)/2:(1280+roi_size)/2]
        warp = cv2.rectangle(warp, ((1280-roi_size)/2, 720-roi_size), ((1280+roi_size)/2,720),(0,255,0),2)
        showImg(warp)
        # cv2.imwrite('road/' +filename, road_roi)

def extract_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    count = 0
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened() and count < video_len - 2):
        ret, img = cap.read()
        printProgressBar(count, video_len, length = 50)
        if ret == True:
            img = cv2.resize(img, (1280,720))
            roi_size = 80
            vertical_displacement = 0
            horizontal_displacement = 0
            px1 = 720-roi_size - vertical_displacement
            px2 = 720 - vertical_displacement
            road_roi = img[px1:px2 , (1280-roi_size)/2:(1280+roi_size)/2]
            # warp = cv2.rectangle(warp, ((1280-roi_size)/2, px1), ((1280+roi_size)/2,px2),(0,255,0),2)
            # showImg(warp)
            cv2.imwrite(material+'/'+material+'_' +str(count) + '.png', road_roi)
            # print "saved "+ str(count) + '.png' + " ..."
            count += 1
    cap.release()

def manual_extract_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    count = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            showImgWithRoi(img)

    cap.release()

manual_extract_video(video_path)
# extract_video(video_path)