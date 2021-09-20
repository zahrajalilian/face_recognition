# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 02:03:00 2021

@author: lenovo.center
"""

import face_recognition
import cv2







# load image to detect
#step1 : read image
# orginal_image= cv2.imread('./images/testing/trump-modi.jpg')
orginal_image= cv2.imread('./images/testing/trump-modi-unknown.jpg')
# cv2.imshow('test', image_to_detect)



"""
new code part 
"""
# load pic and extract face encoding


modi_image = face_recognition.load_image_file("images/samples/modi.jpg")
modi_face_encodings = face_recognition.face_encodings(modi_image)[0] #[0] frist image

#return a list of 128 dementional face encoding'


trump_image = face_recognition.load_image_file("images/samples/trump.jpg")
trump_face_encodings = face_recognition.face_encodings(trump_image)[0] #[0] frist image


#create 2 array
#1st array to save the encoding
known_face_encodings=[modi_face_encodings,trump_face_encodings]



#2end array to hold labels ==>same order of faces name
known_face_names=['Narendra Modi','Donald Trump']



#load an unknown image to identify the faces 

# image_to_recognize = face_recognition.load_image_file('./images/testing/trump-modi.jpg')
image_to_recognize = face_recognition.load_image_file('./images/testing/trump-modi-unknown.jpg')
#fid all the faces encoding in the unknown pic



# find all face location using face_location() func
# model can be cnn or hog  ==>hog faster than cnn
#num_of_times_to_upsample=1 higher and detect more faces
# step2: all faces
all_face_locations=face_recognition.face_locations(image_to_recognize,model='hog')
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)

print('there are {} of faces in this image'.format(len(all_face_locations)))

# get face location 




# step4: loop through faces location and encodings 
for current_face_location,current_face_encodings in zip(all_face_locations,all_face_encodings):
    #step5: print location of each faces ==>split the tuple 
    # top_pos,left_pos,righ_pos,bottom_pos = 
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    
    
    
    
    
    #print('found faces {} at top :{} rigt:{},bottom:{},left:{}'.format(index+1,top_pos
                                                                               #,right_pos,bottom_pos,left_pos))
    
    
    
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
    cv2.rectangle(orginal_image,(left_pos,top_pos),(right_pos,bottom_pos), (255,0,0),2)
    
    #write name below faces
    font =cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(orginal_image,name_of_person, (left_pos,bottom_pos+20), font, 0.5,(255,255,255),1)
    
    
    

 
    cv2.imshow('face identical:',orginal_image)
    
    
"""
we use orginal image

"""
    
