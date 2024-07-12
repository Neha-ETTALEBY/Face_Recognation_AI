import threading
import cv2
from deepface import DeepFace

# Open the first camera of the laptop and set the frame size to 640x480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables
counter = 0
face_match = False
reference_img = cv2.imread("reference_img.jpg")

def check_face(frame):
    global face_match
    try:
        # Use the DeepFace library to verify if the current frame matches the reference image
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        # If there is an error during the verification process, set face_match to False
        face_match = False

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if ret:
        # Check for a face match every 30 frames (to improve performance)
        if counter % 30 == 0:
            try:
                # Run the face verification in a separate thread to avoid blocking the main loop
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # Display the result on the frame
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("video", frame)

    # Wait for the 'q' key to be pressed to exit the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the camera and close all windows
cv2.destroyAllWindows()