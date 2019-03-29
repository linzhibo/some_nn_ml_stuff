import math
import sys
import cv2
import os
import numpy as np
import time
import sys
from datetime import datetime, timedelta
class getRoiApp(object):
    def __init__(self, video_path):
        '''
        trackbar to move roi vertically and horizontally
        third trackbar to choose material to save
        space to skip
        esc to quit
        y to save current roi
        k to keep/stop saving 
        f to switch material
        arrows to move roi
        '''
        self.video_path = video_path
        self.materials = ['asphalt', 'brick', 'cobblestone', 'concrete', 'flagstone', 'mud']
        self.current_material = 'asphalt'
        self.current_time = time.time()
        self.material_index = 0
        self.count = 0
        self.skip = 0
        self.keep_saving = 0
        self.dragging = 0
        self.vertical = 0
        self.horizontal = 0
        self.roi_size = 80
        self.height = 720
        self.width = 1280
        self.video_timer = 0
        self.progress = 0
        self.total_time = 0
        self.total_frame = 0
        self.color = (255,255,255)
        self.timeLinePos = 0
        self.timeLineModified = 0
        self.videoCap = 0
        self.img = 0
        self.timeLineLength = 1000
        

    def nothing(self, x):
        pass
    
    def saveRoi(self, f_name, road_roi):
        cv2.imwrite(f_name, road_roi)
        print 'saving ' +f_name + " ..."
        self.count += 1
    
    def mouseDragging(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.dragging is False:
                self.dragging = True
                self.horizontal = x - self.roi_size/2
                self.vertical = y - self.roi_size/2

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging is True:
                self.horizontal = x - self.roi_size/2
                self.vertical = y - self.roi_size/2
        elif event == cv2.EVENT_MOUSEWHEEL:
            print flags

        else:
            self.dragging = False
        cv2.setTrackbarPos('horizontal', 'img', self.horizontal)
        cv2.setTrackbarPos('vertical', 'img', self.vertical)
    
    def setTimeLine(self, x):
        pos = int(x*self.total_frame/self.timeLineLength)
        self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, self.img = self.videoCap.read()
        if ret == True:
            self.img = cv2.resize(self.img, (1280,720))
            self.updateStatus()
            

    def initWindow(self):
        cv2.namedWindow("img")
        cv2.moveWindow("img", 300,200)
        cv2.createTrackbar('horizontal','img', 0,self.width-self.roi_size, self.nothing)
        cv2.createTrackbar('vertical','img', 0,self.height-self.roi_size, self.nothing)
        cv2.createTrackbar('timeline','img', 0, self.timeLineLength, self.setTimeLine)
        cv2.createTrackbar('material', 'img', 0, 5, self.nothing)
    
    def checkArrowKeys(self, k):
        # 81 left, 82 up, 83 right, 84 down
        move_pixel = 10
        if k == 81:
            cv2.setTrackbarPos('horizontal', 'img', self.horizontal - move_pixel)
        if k == 83:
            cv2.setTrackbarPos('horizontal', 'img', self.horizontal + move_pixel)
        if k == 82:
            cv2.setTrackbarPos('vertical', 'img', self.vertical - move_pixel)
        if k == 84:
            cv2.setTrackbarPos('vertical', 'img', self.vertical + move_pixel)
    
    def checkPlusMinus(self, k):
        if k == 43:
            if self.roi_size < 200:
                self.roi_size += 10
        if k == 45:
            if self.roi_size > 10:
                self.roi_size -= 10
            

    def showMainWindow(self):
        cv2.setMouseCallback("img", self.mouseDragging)
        while(1):
            src  = self.img.copy()
            self.vertical = cv2.getTrackbarPos('vertical', 'img')
            self.horizontal = cv2.getTrackbarPos('horizontal', 'img')
            material = self.materials[cv2.getTrackbarPos('material', 'img')]
            cv2.putText(src, material, (self.roi_size+self.horizontal,self.vertical+self.roi_size/2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.putText(src, 'progress: ' + str('%.3f'%self.progress)+ ' %', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, lineType=cv2.LINE_AA)
            cv2.putText(src, 'time: ' + self.video_timer +' /' + self.total_time, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, lineType=cv2.LINE_AA)
            road_roi = self.img[self.vertical: self.roi_size+self.vertical, self.horizontal: self.roi_size+self.horizontal]
            cv2.rectangle(src, (self.horizontal,self.vertical),(self.roi_size+self.horizontal,self.roi_size+self.vertical), (0,255,0),2)
            cv2.imshow("img", src)
            f_name = material+'/'+str(self.current_time)+'_'+str(self.count) + '.png'
            if self.skip:
                k = cv2.waitKey(1) & 0xFF
                if k == 32:
                    self.skip =0
                return
            if self.keep_saving:
                k = cv2.waitKey(1) & 0xFF
                if k == 107:
                    self.keep_saving = 0
                self.saveRoi(f_name, road_roi)
                return
            k = cv2.waitKeyEx(1) & 0xFF
            # if k != 255:
            #     print k
            self.checkArrowKeys(k)
            self.checkPlusMinus(k)
            # 102 = f
            if k == 102:
                self.material_index = cv2.getTrackbarPos('material', 'img')
                self.material_index += 1
                if self.material_index >= len(self.materials):
                    self.material_index = 0
                cv2.setTrackbarPos('material', 'img', self.material_index)
            # 107 = k
            if k == 107:
                self.keep_saving = 1
            # 121 = y
            if k == 121:
                self.saveRoi(f_name, road_roi)
            # 27 = esc
            if k == 27:
                cv2.destroyAllWindows()
                sys.exit(1)
            # 32 = space
            if k == 32:
                self.skip = 1
                return

    def updateStatus(self):
        crop = self.img[0:100,0:100]
        # print np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        if np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) > 200:
            self.color = (0,0,0)
        else:
            self.color = (255,255,255)
        self.progress = 100*self.videoCap.get(cv2.CAP_PROP_POS_FRAMES)/self.total_frame
        self.video_timer = timedelta(seconds=0.001*self.videoCap.get(cv2.CAP_PROP_POS_MSEC))
        self.video_timer = str(self.video_timer - timedelta(microseconds=self.video_timer.microseconds))

    def run(self):
        self.initWindow()
        self.videoCap = cv2.VideoCapture(self.video_path)
        if (self.videoCap.isOpened()== False): 
            print("Error opening video stream or file")
        self.total_frame = self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_time = self.total_frame/self.videoCap.get(cv2.CAP_PROP_FPS)
        self.total_time = timedelta(seconds=self.total_time)
        self.total_time = str(self.total_time - timedelta(microseconds=self.total_time.microseconds))
        while(self.videoCap.isOpened()):
            ret, self.img = self.videoCap.read()
            if ret == True:
                self.img = cv2.resize(self.img, (1280,720))
                self.updateStatus()
                cv2.setTrackbarPos('timeline', 'img', int(self.progress*0.01 *self.timeLineLength))
                self.showMainWindow()
            if self.progress== 100.0:
                print "reached end. . . "
                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    break
        self.videoCap.release()


video_path = '/home/zlin/Downloads/30 Mins Treadmill Workout Scenery. Virtual Scenery For Exercise Machine (Cotswolds UK).mp4'
app = getRoiApp(video_path)
app.run()
