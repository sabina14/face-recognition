import cv2
import numpy as np
import os
import face_recognition
import datetime as datetime





#encodeListKnown=faceEncoding(images)

# Open the input movie file

#input_movie = cv2.VideoCapture("input.mp4")
input_movie = cv2.VideoCapture("video.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output1.avi', fourcc, 25, (1280, 720))






# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file("Train\Abdul_Kalam.jpg")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]


al_image = face_recognition.load_image_file("Train\PM_Modi.jpg")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
   al_face_encoding
]



# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #print(frame)
    #rgb_frame = frame[:, :, ::-1]
    #print(rgb_frame)
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Abdul_Kalam"
        elif match[1]:
            name = "PM_Modi"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
