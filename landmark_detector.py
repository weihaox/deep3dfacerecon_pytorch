#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   landmark_detector.py
@Time    :   2023/03/22 22:48:50
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   detect 5 facial landmarks for images
'''


import dlib
import cv2
import os

detector = dlib.get_frontal_face_detector()
predictor_model = "shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_model)

# read_path = 'datasets/synthestic_3d_face/examples'
# read_path = 'datasets/synthestic_3d_face/img'
# read_path = 'datasets/ffhq-256-kaggle/img'
# read_path = 'datasets/celeba-hq-256/img'
read_path = 'datasets/celeba-hq-256/img_missing'
save_path = os.path.join(read_path, 'detections')

if not os.path.exists(save_path):
    os.makedirs(save_path)

for filename in os.listdir(read_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load image
        image = cv2.imread(os.path.join(read_path, filename))
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = detector(gray)
        # Loop through faces
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            # Save landmarks to file
            output_filename = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(save_path, output_filename), "w") as f:
                for i in range(5):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    f.write("{:.2f} {:.2f}\n".format(x, y))