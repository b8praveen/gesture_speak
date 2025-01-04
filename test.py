import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the webcam, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Parameters
offset = 20
imgSize = 300

# Labels for classification
labels = ["Hello", "Okay", "Thank You", "Yes", "No", "Stop"]

while True:
    # Capture frame from the webcam
    success, img = cap.read()
    
    # Check if frame was successfully captured
    if not success:
        print("Failed to capture frame from webcam. Exiting...")
        break  # Exit the loop if frame capture fails
    
    # Make a copy of the original frame for output
    imgOutput = img.copy()
    
    # Find hands in the frame
    hands, img = detector.findHands(img)
    
    # If hands are detected, proceed with processing
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure the cropping coordinates are within image bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        
        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop the hand region from the image
        imgCrop = img[y1:y2, x1:x2]
        
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
        
        # Make prediction using the classifier
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(f"Prediction: {prediction}, Index: {index}")
        
        # Draw rectangle and label on the output image
        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
        
        # Display images
        cv2.imshow('Cropped Image', imgCrop)
        cv2.imshow('White Image', imgWhite)

    # Display the main image
    cv2.imshow('Webcam', imgOutput)
    
    # Check for key press and exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
