#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pygame,dlib,time,cv2,os
from imutils.video import VideoStream
from imutils import face_utils


# In[13]:


shape_predictor="shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


# In[23]:


vs = VideoStream(src=0).start()
time.sleep(2.0)


# In[24]:


count=0
p1=[(0,0)]*68
p2=[(0,0)]*68
d=[(0,0)]*68
dist_smilo=0
diff_chx,diff_chy=0,0
pid=0
count_smile,count_eact,count_be=0,0,0


# In[22]:


while True:
        
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
        
    diff_smile=0
    diff_ang=0
    diff_leye=0
    diff_eye=0
    diff_reye=0
    diff_up=0
    diff_change=0
        
    if(count%2==0):
        p1=p2
        p2=[(0,0)]*68
        d=[(0,0)]*68
    
    cv2.imshow("frame",frame)
    
    x49=0
    y49=0
    x55=0
    y55=0
    
    #print("Count smile ",count_smile)
    
    s=0
    
    for rect in rects:
        shape=predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)
        i=1
        
        x1,y1,w,h = 0,0,0,0
        count=count+1
        
        for (x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,255,0),-1)
            if(i):
                cv2.putText(frame, str(i), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            if (i==49):
                x49=x
                y49=y
            elif(i==55):
                x55=x
                y55=y
                dist_smile=((x49-x55)**2+(y49-y55)**2)**0.5
                #print('dist_smile',dist_smile)
                diff_smile = (dist_smile)-dist_smilo
                if diff_smile<0:
                    diff_smile*=-1
                #print('diff_smile',diff_smile)
                #print('dist_smilo',dist_smilo)
                if count==1 or diff_smile>15:
                    dist_smilo=dist_smile
                if diff_smile<6:
                    dist_smilo = (dist_smilo+dist_smile)//2
            if (diff_smile>10 and dist_smile>60 and dist_smilo>55 and count!=1):
                #cv2.putText(frame,'Smile', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                s=1
                #cv2.imshow("selfie1", frame)
            
            i=i+1
    
    if (s==1):
        count_smile = count_smile+1
        print("Smile ",count_smile)
        s=0
    
        
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
        
    if key == 27:
        break

VideoStream(src=0).stop()
cv2.destroyAllWindows()


# In[ ]:




