import cv2 as cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

import numpy as np
import scipy.misc

#from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model

#from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,Dropout,MaxPool2D,Flatten

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

base_mod=Sequential()

base_mod.add(Conv2D(filters=32,kernel_size=(5,5),strides=2,activation='relu'))

base_mod.add(MaxPool2D(pool_size=(3,3)))

base_mod.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
base_mod.add(tf.keras.layers.ZeroPadding2D(
    padding=(2, 2), data_format=None
))

base_mod.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
base_mod.add(tf.keras.layers.ZeroPadding2D(
    padding=(1, 1), data_format=None
))


base_mod.add(MaxPool2D(pool_size=(3,3),strides=2))
base_mod.add(Dropout(0.15))
#base_mod.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))

#base_mod.add(MaxPool2D((2,2)))     cause issse kaafi chots ho jata hai input matrix so kkaaf info loss atleast 1/5 of 
#height and width rehna chahiye before flattening

base_mod.add(Flatten())

base_mod.add(Dense( units=1500,
    activation='relu'))
base_mod.add(Dropout(0.15))



base_mod.add(Dense( units=1500,
    activation='relu'))
base_mod.add(Dropout(0.15))

base_mod.add(Dense( units=600,
    activation='relu'))
base_mod.add(Dropout(0.15))

base_mod.add(Dense( units=100,
    activation='relu'))
base_mod.add(Dropout(0.15))

base_mod.add(Dense( units=3,
    activation='softmax'))
#base_mod.load_weights('D:/opencv/trainv2-weights-2.tf')
new_model = tf.keras.models.load_model(r'my_model.h5')

new_model.summary()

#print(base_mod.summary())'''
cap = cv2.VideoCapture(0)
i=0
capture=[np.zeros((480, 640, 3))]
j=0
while(True):

  ret,frame = cap.read()
  #qprint(frame.shape)
  if ret == True:
    
    cv2.imshow('Frame', frame)
    cv2.imshow('captured',capture[i])
    k=cv2.waitKey(1)
 
    if k == ord('q'):
      break
    if k== ord('c'):
        i+=1
        capture.append(frame)
        img=cv2.resize(capture[i],(256,256))
        cv2.imwrite(f'minor-dataset/C/{j}.jpg',img)
        j+=1
        #img=cv2.resize(capture[i],(100,100))
        img=cv2.resize(capture[i],(64,64))
        print(frame.shape)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=np.array(img).reshape(1,64,64,3)
        img=img/255
        print(np.round(new_model.predict(img),1))
        
        

  else: 
    break

cap.release()
   

cv2.destroyAllWindows()