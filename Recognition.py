#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:51:20 2020

@author: djkim9031
"""


import cv2, time
import numpy as np
import face_recognition
import imutils
import os
from threading import Thread
from datetime import datetime

path = 'attendance/data'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    if cl.endswith('.jpg'):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

def markAttendance(name):
    with open('attendance/info.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString}')
        
    
encodeListKnown = findEncodings(images)
print('Encoding Complete')

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.count = 0
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                
                faceFrame = face_recognition.face_locations(self.frame)
                encodeFrame = face_recognition.face_encodings(self.frame,faceFrame)
                
                for encodeFace, faceLoc in zip(encodeFrame,faceFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIdx = np.argmin(faceDistance)
                    
                    if matches[matchIdx]:
                        name = classNames[matchIdx].upper()
                        markAttendance(name)
                        cv2.rectangle(self.frame,(faceLoc[1],faceLoc[2]),(faceLoc[3],faceLoc[0]),(0,0,255),2)
                        cv2.putText(self.frame,f'{name}',(faceLoc[3],faceLoc[0]+10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        

            time.sleep(self.FPS)

    def show_frame(self):
        cv2.imshow('Trailer', self.frame)
        cv2.waitKey(self.FPS_MS)

threaded_camera = ThreadedCamera('attendance/trailer.mov')
while True:
    try:
        threaded_camera.show_frame()
        #pass
    except AttributeError:
        pass

