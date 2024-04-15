import cv2
import numpy as np
import face_recognition
import os
import pickle
from imutils import paths

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 75)
    height = int(frame.shape[0] * percent/ 107)
    
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def recognizer(args):
    print('Running ... ')
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    
    

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # load the known faces and embeddings saved in last file
    data = pickle.loads(open(args.encoded_path, "rb").read())

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image frbom BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #     rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(data['images'], face_encoding,tolerance=0.43)
                name = "UNKNOWN"

                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(data['images'], face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = data['names'][best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#             rect, frame = cap.read()
        frame75 = rescale_frame(frame, percent=150)
        cv2.imshow('frame75', frame75)
            
        # Display the resulting image
#         cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Commands for facial recognition')
    parser.add_argument('--encoded_path',default='encoded_images',type=str)
    args = parser.parse_args()
    
    recognizer(args)
