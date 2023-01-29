#----------------------------UDHAM SINGH ------------------------------------
#---------------------------MDU UNIVERSITY-------------------------------------
#------------------------FACE MASK DETECTION---------------------------------

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import winsound
# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model=tf.keras.models.load_model('mymodel.h5')
# Start the webcam
video_capture = cv2.VideoCapture(0)

video_capture.set(3,1650)
video_capture.set(4,1050)



while True:
    # Capture a frame from the webcam
    _, frame = video_capture.read()
    print(video_capture.get(3), video_capture.get(4))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        # Assume that the person is not wearing a mask
        mask = False
        
        # Crop the face region and apply a mask detector model
        face_region = cv2.resize(gray[y:y+h, x:x+w], (150, 150),interpolation = cv2.INTER_LINEAR)
        face_region = np.expand_dims(face_region, axis=0) # to add the batch dimension


        # You need to use your own mask detector model here
        mask = model.predict(face_region) 
        
        # Draw a rectangle around the face
        if mask:
            color = (0, 255, 0) # Green
        else:
            color = (0, 0, 255) # Red
            winsound.Beep(2000, 500)  # play a beep sound
            cv2.putText(frame, "Please remove the mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), tuple(color), 2)

    # Display the frame
    cv2.imshow("Video", frame)
    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()


