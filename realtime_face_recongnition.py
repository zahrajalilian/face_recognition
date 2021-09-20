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


"""
new part

"""


modi_image = face_recognition.load_image_file("images/samples/modi.jpg")
modi_face_encodings = face_recognition.face_encodings(modi_image)[0] #[0] frist image

#return a list of 128 dementional face encoding'


trump_image = face_recognition.load_image_file("images/samples/trump.jpg")
trump_face_encodings = face_recognition.face_encodings(trump_image)[0] #[0] frist image




zahra_image = face_recognition.load_image_file("images/samples/zahra.jpg")
zahra_face_encodings = face_recognition.face_encodings(zahra_image)[0] #[0] frist image



#create 2 array
#1st array to save the encoding
known_face_encodings=[modi_face_encodings,trump_face_encodings,zahra_face_encodings]



#2end array to hold labels ==>same order of faces name
known_face_names=['Narendra Modi','Donald Trump',"Zahra Jalilian"]







# step two
#initialize empty array for face locations,encodings,names

all_face_locations = []
all_face_encodings =[]
all_face_names=[]



#step 3:
    #create outer loop through video

while True:
    # step 4:get single frame of video as image
    ret ,current_frame = webcam_video_stream.read()
    # step 5: resize to smaller size to prossess faster
    current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces
    all_face_locations=face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
    
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    
    all_face_names=[]
    
    
    # step4: loop through faces location and encodings 
    for current_face_location,current_face_encodings in zip(all_face_locations,all_face_encodings):
        #step5: print location of each faces ==>split the tuple 
        # top_pos,left_pos,righ_pos,bottom_pos = 
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        
        top_pos= top_pos*4
        right_pos=right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        
        """
        new
        """
        #compare faces and get mached faces
        all_maches= face_recognition.compare_faces(known_face_encodings, current_face_encodings)
        #initialize name str 
        name_of_person="Unknown Face"
        
        
        
        #use the frist match and get the name
        
        if True in all_maches :
            first_match_index = all_maches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        
        
        #draw rectangle around the face
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos), (255,0,0),2)
        
        #write name below faces
        font =cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,name_of_person, (left_pos,bottom_pos+20), font, 0.5,(255,255,255),1)
        
        
        
    
     
        cv2.imshow('face identical:',current_frame)
        
        
        
        
    
    
    
    
    
    
    
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#break the video when pres q
#releasethe camera and cllose the windows
webcam_video_stream.release()
cv2.destroyAllWindows()
