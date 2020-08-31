import os
import cv2
from tqdm import tqdm

animals = ["Bald eagle", "Black bear", "Bobcat", "Canada lynx", "Columbian black-tailed deer", "Cougar", "Coyote", "Deer", "Elk", "Gray fox",
         "Gray wolf", "Nutria", "Raccoon", "Raven", "Red fox", "Ringtail", "Sea lions", "Seals", "Virginia opossum"]


train_pics = "D:/Data Science Proj Datasets/oregon_wildlife_train_set"
for animal in tqdm(os.listdir(train_pics)):
    for filename in tqdm(os.listdir(f"{train_pics}/{animal}")):
        print(filename)
        img_arr = cv2.imread(os.path.join(train_pics, filename))
        # encoding = face_recognition.face_encodings(img)
        # known_labels.append(filename)
        # known_faces.append(encoding)

# train_faces = "C:/Users/User/Desktop/NTU Data Science/Facial Recognition/Train Faces/Charlize Theron/19.jpg"
# img = face_recognition.load_image_file(train_faces)


