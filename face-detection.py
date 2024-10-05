import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

time.sleep(2)
print('Starting Face Detection...')


saved_faces = set()

while True:
    
    ret, frame = cap.read()

    # print('1')
    
    # if ret:
    #     cv2.imshow('Webcam Feed', frame)
    # else:
    #     print("Failed to capture frame")

    # Convert the frame to grayscale (needed for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale Frame', gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
     
     # Save detected face
        # face = frame[y:y+h, x:x+w]
        # cv2.imwrite(f'detected_face_{time.time()}.jpg', face)
        face_coords = (x, y, w, h)
        
 
   
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
