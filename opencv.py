# Importing the Computer Vision Module
import cv2

# Determining the Classifier for Face Detection
cascade1 = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Starting WebCam
cap = cv2.VideoCapture(0)

# Infinite Loop for capturing every frame
while True:

    # Reading the Frame
    _, imgeg = cap.read()
    '''.read() function returns 2 outputs :
        1 is the Boolean value of connecting with a webcam
        2nd is the actual frame which is captured'''

    # imgeg = cv2.imread('work.jpg')

    # Converting Captured image into Grayscale for Classifier to Detect
    gray = cv2.cvtColor(imgeg, cv2.COLOR_BGR2GRAY)

    # Detecting the Faces
    faces = cascade2.detectMultiScale(gray, 1.2, 9)
    eyes = cascade1.detectMultiScale(gray, 1.2, 9)
    # Creating rectangle around the detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(imgeg, (x, y), (x + w, y + h), (0, 0, 250), 2)
        # Putting Text over the recognised face
        cv2.putText(imgeg, "Face Detected", (x-10, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    for (x, y, w, h) in eyes:
        cv2.rectangle(imgeg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ''' .rectangle() function takes 5 arguments :
            - the object to be drawn around
            - x and y coordinates to begin the rectangle
            - corner points of the rectangle
            - color of the rectangle
            - thickness of the rectangle'''

    # Showing Real Time Frames captured
    cv2.imshow('img', imgeg)

    # Exiting with Esc keypress
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Releasing the WebCam
cap.release()
