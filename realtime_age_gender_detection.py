# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 03:00:28 2021

@author: lenovo.center
"""

import face_recognition
import cv2

#step one

#get webcam(default one,1,2,3  aditionals atached cams)
webcam_video_stream=cv2.VideoCapture(0)


# step two
#initialize empty array for face locations

all_face_locations = []

#step 3:
    #create outer loop through video

while True:
    # step 4:get single frame of video as image
    ret ,current_frame = webcam_video_stream.read()
    # step 5: resize to smaller size to prossess faster
    current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces
    all_face_locations=face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    for index ,current_face_location in enumerate(all_face_locations):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        #change the position maginitude to fit the actual size video frame
        top_pos= top_pos*4
        right_pos=right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        print('found faces {} at top :{} rigt:{},bottom:{},left:{}'.format(index+1,top_pos
                                                        
                ,right_pos,bottom_pos,left_pos))
        # extract the face from the frame blur it paste it back to the frame
        #slice the current image
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        
        
        
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
       
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,gender+" "+age+" "+"years", (left_pos,bottom_pos+20), font, 0.5,(0,255,0),1)
    # show faces with rectangle
    cv2.imshow('webcam video',current_frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#break the video when pres q
#releasethe camera and cllose the windows
webcam_video_stream.release()
cv2.destroyAllWindows()
