print("Importing libraries..")
import cv2
import os
import face_recognition
from tqdm import tqdm
import pickle
import numpy as np
print("Libraries imported!")

count = 0

#train_faces = "C:/Users/User/Desktop/NTU Data Science/Facial Recognition/Train Faces"
test_faces = "C:/Users/User/Desktop/NTU Projects/Facial Recognition/Test Faces"
tol = 0.6
frm_thik = 3
font_thik = 2
MODEL = "hog"

#known_faces = []
#known_labels = []

#bringing in the known faces
#for name in os.listdir(train_faces):
    #print("Celeb: ", name)
    #for filename in os.listdir(f"{train_faces}/{name}"):
        #print("File name: ", filename)
        #img = face_recognition.load_image_file(f"{train_faces}/{name}/{filename}")
        #encoding = face_recognition.face_encodings(img)[0]
        #known_labels.append(name)
        #known_faces.append(encoding)

### loading encoded faces
known_faces = list(np.load('C:/Users/User/Desktop/NTU Projects/Facial Recognition/will.i.am/will.i.am/known_faces.npy'))
### and the known labels
with open("C:/Users/User/Desktop/NTU Projects/Facial Recognition/will.i.am/will.i.am/known_labels.pickle", "rb") as f:
    known_labels = pickle.load(f)

#processing the unknown faces
for filename in os.listdir(test_faces):
    unkwn_img = face_recognition.load_image_file(f"{test_faces}/{filename}")
    loc = face_recognition.face_locations(unkwn_img, model = MODEL)
    encoding_unkwn = face_recognition.face_encodings(unkwn_img, loc)
    opencv_img = cv2.cvtColor(unkwn_img, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encoding_unkwn, loc):
        match, ress = face_recognition.compare_faces(known_faces, face_encoding, tol)
        if match:
            match = known_labels[ress.argmin()]
            print(f"Match found: {match}")

            #drawing rectangle
            tp_left = (face_location[3], face_location[0])
            btm_right = (face_location[1], face_location[2])
            col_rec = [255, 0, 0]
            cv2.rectangle(opencv_img, tp_left, btm_right, col_rec, frm_thik)
            #a smaller rectangle to print the label
            white_txt = (200,200,200)
            top_left = (face_location[3], face_location[2])
            bot_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(opencv_img, top_left, bot_right, col_rec, cv2.FILLED)
            cv2.putText(opencv_img, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_txt, font_thik)
            save_folder = 'results3'
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(f'results3/{count}.jpg', opencv_img)
            count+=1
