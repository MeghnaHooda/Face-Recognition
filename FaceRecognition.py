
# coding: utf-8

# In[2]:


import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("trainner.yml")

cap = cv2.VideoCapture(0)

while(True):
    #Capture frame by frame
    ret, frame = cap.read()
    
    #converting the image to gray and finding the face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=5.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        img_item = "my-image.png"
        
        id_, conf = recognizer.predict(roi_color)
        if conf>=45 and conf<= 85:
            print(id_)
        
        
        cv2.imwrite(img_item, roi_color)
        
        color = (255, 0, 0)#BGR 0-255
        stroke=2
        endx = x + w
        endy = y + h
        cv2.rectangle(frame,(x,y),(endx, endy), color, stroke)
        
    
    #Display the resulting frame
    cv2.imshow('myframe',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

