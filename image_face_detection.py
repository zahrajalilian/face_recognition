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
    cv2.imshow('face No:'+str(index+1),current_face_image)
    
    
    
