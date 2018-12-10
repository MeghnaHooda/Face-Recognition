
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()

current_id = 0
label_ids = {}
x_train = []
y_label = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpeg'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label]= current_id
                current_id +=1
            id_ = label_ids[label]
            #print(label_ids)
            
            
            pil_image = Image.open(path).convert("L")
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=5.5, minNeighbors=5)
            
            for(x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id_)
#print(x_train)
#print(y_label)
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids,f)
    
recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")

