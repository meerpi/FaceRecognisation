import cv2 as cv
import os
import numpy as np

vid = cv.VideoCapture(0)
main_path = 'pics'
if not vid.isOpened():
    print("Error: Camera cannot be accessed")
    exit()

# Load Haarcascade XML file for face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create the LBPH recognizer and load the trained model
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Ensure this file exists and contains trained data

font = cv.FONT_HERSHEY_SIMPLEX  # Font for text on the video frame

# Dynamically create the list of names based on folder names
names = []  # This will store the folder names as labels for recognized IDs

# Map folder names to numeric IDs based on the order of folders in 'main_path'
for folder_name in os.listdir(main_path):
    folder_path = os.path.join(main_path, folder_name)
    
    if os.path.isdir(folder_path):  # Check if it's a folder (person's name)
        print(f"Adding {folder_name} to known faces")
        names.append(folder_name)  # Add folder name (person's name) to the list

# Loop through the video frames
while True:
    ret, frame = vid.read()
    if not ret:
        print("Error: Unable to capture video")
        break

    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)  # Detect faces

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (10, 159, 255), 2)  # Draw rectangle around face

        # Recognize the face
        id, confidence = recognizer.predict(grey[y:y+h, x:x+w])

        # Check confidence level: lower confidence means higher certainty
        if confidence < 100:
            # Get the name from the 'names' list based on the predicted ID
            name = names[id] if id < len(names) else "Unknown"
            confidence_text = f"{100 - confidence:.0f}%"  # Confidence percentage
        else:
            name = "Unknown"
            confidence_text = f"{100 - confidence:.0f}%"

        # Print the result (recognized name and confidence level)
        print(f"Detected: {name}, Confidence: {confidence_text}")

        # Display the name and confidence on the frame
        cv.putText(frame, f"{name} ({confidence_text})", (x+5, y-5), font, 1, (255, 255, 255), 2)

        # Check if confidence is over 60%, if so, stop the camera
        if (100 - confidence) > 50:
            print(f"Face recognized with {name}, stopping camera.")
            vid.release()  # Stop the video capture
            cv.destroyAllWindows()  # Close the OpenCV windows
            exit()  # Exit the program

    # Flip the frame horizontally and display it
    frame = cv.flip(frame, 1)
    cv.imshow('Face Recognition', frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
vid.release()
cv.destroyAllWindows()
