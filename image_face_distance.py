# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 02:03:00 2021

@author: lenovo.center
"""

import face_recognition
import cv2





image_to_recognize_path= './images/testing/trump.jpg'





# load image to detect
#step1 : read image
orginal_image= cv2.imread(image_to_recognize_path)

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
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
#fid all the faces encoding in the unknown pic


image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0] #[0] frist image


#load and use face_distance
"""
finding face distance
"""
face_distances = face_recognition.face_distance(known_face_encodings,image_to_recognize_encodings)

#loop through everu face distance
for i,face_distance in enumerate(face_distances):
    # print("the calculated face distance is {:.2} from sample image {}".format(face_distance, known_face_names[i]))

    print("the matching  distance is {} from sample image {}".format(round(((1-float(face_distance))*100),2), known_face_names[i]))




""" 

face distance logical calculation


"""








