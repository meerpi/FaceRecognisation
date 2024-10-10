import cv2 as cv
import os
import time

name = input("Enter your name: ")
save_interval = 0.1  # 10 times per second
last_save_time = time.time()  # Initialize the last save time

while (not name.isalpha()):
    print("Invalid input! Please try again.")
    name = input("Enter your name: ")

vid = cv.VideoCapture(0)# camera 0 is switched on vid IS Capturing the video from the camera
if not vid.isOpened():
    print("THE CAMERA CANNOT BE OPENED")
    exit()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')# load haarcascade xml file
path = os.path.join('pics/',name)
os.mkdir(path)
os.makedirs(path, exist_ok=True)
j = 1
w = 0
x = 0
while True:
    x,y,w,h = 0,0,0,0
    ret, frame = vid.read()# ret gives true or false to indicate if the frame is read or not
    if not ret:
        print("Error....")
        break
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)# converts the image to grey scale as the haarcascade works on grey scale images
    faces = face_cascade.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 6)#haarcascade function to detect faces
    for i in faces:# i is a tuple of 4 elements x,y,w,h down for each
        x,y,w,h = i
        cv.rectangle(frame, (x,y), (x+w ,y+h), (10,159,255),3 )#(frame, start point, end point, color, thickness)
        
    frame = cv.flip(frame,1)#flip the frame other wise mirror image will be shown
    cv.imshow('frame',frame)
    curr_time = time.time()
    if len(faces) > 0 and (curr_time - last_save_time) >= save_interval:
        cv.imwrite(os.path.join(path, name+" "+str(j)+" "+'.jpg'), grey[y:y+h, x:x+w])# only a sliced part of each frame is saved containlg only the face
        j += 1
        last_save_time = curr_time
    if cv.waitKey(1)&0xFF == ord('q'):# press q to exit the loop or close camera
        break
    
vid.release()#close the camera
cv.destroyAllWindows()#close the window

    
