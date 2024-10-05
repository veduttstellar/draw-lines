import cv2
import face_recognition
import os
import time

# Load known faces
known_face_encodings = []
known_face_names = []

# Directory containing known faces
known_faces_dir = 'known_faces'

# Load each known face
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_dir):  # Check if it's a directory
        for filename in os.listdir(person_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Load the image file
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)
                # Encode the face
                face_encoding = face_recognition.face_encodings(image)
                if face_encoding:  # Check if face encoding was found
                    known_face_encodings.append(face_encoding[0])
                    known_face_names.append(person_name)  # Use folder name as the person's name

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

cap = cv2.VideoCapture(0)

time.sleep(2)
print('Starting Face Detection...')

while True:
    ret, frame = cap.read()
    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare each face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the first known face that matches
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Draw rectangles around the faces and label them
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
