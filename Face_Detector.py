import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haarcascade_fraontalface_default.xml)
trained_face_data = cv2.CascadeClassifier('frontal_face.xml')

# Choose an image to detect faces
# img = cv2.imread('stefhan_damon.jpg')


# To capture video from webcam
webcam = cv2.VideoCapture(0)  # 0 is for default camera
# webcam = cv2.VideoCapture('deadpool.mp4')   video



# Iterate forever over frames
while True:

    # Read the curernt frames
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces  detectMultiScale will detect faces of various size of images, composition of eyes,nose etc and will give coordinates of the rectangle
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles

    # (sourceImg,(top left x and y),width,height,color(B,G,R),Thickness of rectangle)
    # cv2.rectangle(img,(123,153),(123+326,153+326),(0,255,0),2)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                    (0, 255, 0), 3)

    print(face_coordinates)
    cv2.imshow('Robert Downey Jr.', frame)
    key=cv2.waitKey(1)  # wait for 1 millisec and go for next frame

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()

# (x,y,w,h) = face_coordinates[0]
"""


# To show the image
cv2.imshow('Robert Downey Jr.', img)  # imshow is image show
cv2.waitKey()

print("Code Completed")
"""
