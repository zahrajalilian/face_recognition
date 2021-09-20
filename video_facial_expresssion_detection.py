# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 03:00:28 2021

@author: lenovo.center
"""
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
import cv2
#step one

#get webcam(default one,1,2,3  aditionals atached cams)
webcam_video_stream=cv2.VideoCapture('images/testing/modi.mp4')




# step two: initialize model and load weights

face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json",'r').read())    
#load weight into model
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#list of emotion labels ==>in this order
  
emotions_label = ('angry','disgust','fear','happy','sad','supries','neutral')  

  
    
    
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
        # extract the face from the frame blur it paste it back to the frameq
        #slice the current image
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        
        # slice the image and draw a rectangle around location
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        
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
        cv2.putText(current_frame,emotion_label, (left_pos,bottom_pos), font, 0.5,(255,255,255),1)
        
        
        
        
    # show faces with rectangle
    cv2.imshow('webcam video',current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#break the video when pres q
#releasethe camera and cllose the windows
webcam_video_stream.release()
cv2.destroyAllWindows()
