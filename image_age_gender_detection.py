# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 02:03:00 2021

@author: lenovo.center
"""

import face_recognition
import cv2



# load image to detect
#step1 : read image
image_to_detect= cv2.imread('./images/testing/trump-modi.jpg')
# cv2.imshow('test', image_to_detect)
# 

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
    
 
    """
    new gender
    """
    #bellow calculated using numpy.mean()
    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)
    # create blub of current flace slice
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image,1,(227,227),AGE_GENDER_MODEL_MEAN_VALUES,swapRB=False)
    #declaring label
    gender_label_list = ['Male','Female']
    #declaring the file path
    gender_protext = "dataset/gender_deploy.prototxt"
    gender_caffemodel = "dataset/gender_net.caffemodel"
    #declare the model
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    #giving input to model
    gender_cov_net.setInput(current_face_image_blob)
    #get the prediction from model
    gender_predictions = gender_cov_net.forward()
    #
    gender = gender_label_list[gender_predictions[0].argmax()]              
    
    
    
    
    
    """
    age new
    """
    age_label_list = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38,43)','(48-53)','(60-100)']
    age_protext = 'dataset/age_deploy.prototxt'
    age_caffemodel = 'dataset/age_net.caffemodel'
    
      #declare the model
    age_cov_net = cv2.dnn.readNet(age_caffemodel,age_protext)
    #giving input to model
    age_cov_net.setInput(current_face_image_blob)
    #get the prediction from model
    age_predictions = age_cov_net.forward()
    #
    age = age_label_list[age_predictions[0].argmax()]              
    
    
    

    
    
    
    
    # slice the image and draw a rectangle around location
   
    cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect,gender+" "+age+" "+"years", (left_pos,bottom_pos+20), font, 0.5,(0,255,0),1)
# show faces with rectangle
cv2.imshow('age gender',image_to_detect)


    
    
