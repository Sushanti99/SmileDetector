#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import dlib
from imutils.video import VideoStream
from imutils import face_utils
import matplotlib.pyplot as plt


# In[ ]:


shape_predictor="shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


# In[ ]:


import time
vs = VideoStream(src=0).start() 
time.sleep(2.0)

while True:
        
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    
    rects = detector(frame,0)
    
    for rect in rects:
        
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        x,y,w,h = face_utils.rect_to_bb(rect)
        t = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
        
        right = frame[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]
        left = frame[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]
        
        cv2.rectangle(frame, (shape[54][0],shape[29][1]), (shape[12][0], shape[33][1]), (0,0,255), 1)
        cv2.rectangle(frame, (shape[4][0], shape[29][1]), (shape[48][0], shape[33][1]), (0,0,255), 1)       
        

        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        
    if key == 27:
        break

VideoStream(src=0).stop()
cv2.destroyAllWindows()

