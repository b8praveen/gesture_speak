import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize the webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
counter = 0

# Ensure the folder path exists
folder = r"C:\Desktop\Mini Project\Sign-Language-detection\No"
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop the hand region from the image
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        
        aspectRatio = h / w
        
        # Resize and paste the cropped image onto the white image
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
        
        # Display images
        cv2.imshow('Cropped Image', imgCrop)
        cv2.imshow('White Image', imgWhite)

    # Display the main image
    cv2.imshow('Webcam', img)
    
    # Save the image when 's' key is pressed
    if cv2.waitKey(1) == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image {counter} saved.")


























