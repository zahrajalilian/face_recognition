# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 02:03:00 2021

@author: lenovo.center
"""
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import cv2



# load image to detect
#step1 : read image
image_to_detect= cv2.imread('./images/testing/trump-modi.jpg')
# cv2.imshow('test', image_to_detect)
# 





# step two: initialize model and load weights

face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json",'r').read())    
#load weight into model
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#list of emotion labels ==>in this order
  
emotions_label = ('angry','disgust','fear','happy','sad','supries','neutral')  

  
    
  
    
# find all face location using face_location() func
# model can be cnn or hog  ==>hog faster than cnn
#num_of_times_to_upsample=1 higher and detect more faces
# step2: all faces
all_face_locations=face_recognition.face_locations(image_to_detect,model='hog')


print('there are {} of faces in this image'.format(len(all_face_locations)))

# get face location 




# step4: loop through faces 
for index ,current_face_location in enumerate(all_face_locations):
    #step5: print location of each faces ==>split the tuple 
    # top_pos,left_pos,righ_pos,bottom_pos = 
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('found faces {} at top :{} rigt:{},bottom:{},left:{}'.format(index+1,top_pos
                                                                               ,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    # slice the image and draw a rectangle around location
    cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    
    """
    new part
    emotion detection 
    """
    #preprocess inut convert to image 
    #convert to grayscale
    # 
    current_face_image=cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
    # resize to 48*48 px size(h,w)
    current_face_image=cv2.resize(current_face_image,(48,48))
    #convert pil image to numpy array
    img_pixles=image.img_to_array(current_face_image)
    #expend the shape of arry into single row multi colunmns
    img_pixles = np.expand_dims(img_pixles,axis=0)
    #pixles are in range [0,255].normalize all pixles inscale [0,1]  
    img_pixles /= 255
    
    """
    predict emotion
    """
    #########
    # the prediction model for all 7 expression
    exp_predictions=face_exp_model.predict(img_pixles)
    # find max indexted predictions value (0 till 7)
    max_index= np.argmax(exp_predictions[0])
    #get corresponding label from emotional label
    emotion_label = emotions_label[max_index]
    
    
    #display the name as text in image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect,emotion_label, (left_pos,bottom_pos), font, 0.5,(255,255,255),1)
    
    
    
    
# show faces with rectangle
cv2.imshow('image face motions',image_to_detect)

    
    
