import cv2
import os
import numpy as np

# Using LBPH(Local Binary Patterns Histograms) recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
main_path = 'pics'  # The main directory where each folder represents a person

# Function to read the images in the dataset, convert them to grayscale values, return samples
def getImagesAndLabels(main_path):
    faceSamples = []
    ids = []
    current_id = 0  # To assign unique numeric IDs to each person (based on folder name)
    label_dict = {}  # Dictionary to map folder names to numeric labels

    # Loop over each folder in the main directory
    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):  # Check if it's a folder (person's name)
            print(f"Processing folder: {folder_name}")

            # Assign a unique ID to the person if not already done
            if folder_name not in label_dict:
                label_dict[folder_name] = current_id
                current_id += 1

            # Loop over each image in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".jpg"):  # Only process JPG images
                    print(f"Processing image: {file_name}")
                    img_path = os.path.join(folder_path, file_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    faces = face_detector.detectMultiScale(img)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img[y:y+h, x:x+w])  # Add the detected face to samples
                        ids.append(label_dict[folder_name])  # Use the folder name as the label

    return faceSamples, ids, label_dict

# Function to train the recognizer
def trainRecognizer(faces, ids):
    recognizer.train(faces, np.array(ids))
    # Create the 'trainer' folder if it doesn't exist
    if not os.path.exists("trainer"):
        os.makedirs("trainer")
    # Save the model into 'trainer/trainer.yml'
    recognizer.write('trainer/trainer.yml')

print("\n Training faces. It will take a few seconds. Wait ...")
# Get face samples and their corresponding labels
faces, ids, label_dict = getImagesAndLabels(main_path)

# Train the LBPH recognizer using the face samples and their corresponding labels
trainRecognizer(faces, ids)

# Print the number of unique faces trained and the person-label mapping
num_faces_trained = len(set(ids))
print(f"\n {num_faces_trained} faces trained. Exiting Program.")
print(f"Label mapping: {label_dict}")
