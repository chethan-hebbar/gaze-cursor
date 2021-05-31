import cv2 
import dlib 
import numpy as np
import os
import pyautogui as cursor
import operator
from imutils.video import FileVideoStream
import time
import imutils

# loading our face-detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
getLandmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# function to get the midpoint between two points
def getMidPoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

# function returns the position of center of iris and position of center of the eye
def getDiff(points, landmarks, image, gray_image):

    # getting the eye with landmarks
    eye_region = np.array([(landmarks.part(points[0]).x, landmarks.part(points[0]).y),
                        (landmarks.part(points[1]).x, landmarks.part(points[1]).y),
                        (landmarks.part(points[2]).x, landmarks.part(points[2]).y),
                        (landmarks.part(points[3]).x, landmarks.part(points[3]).y),
                        (landmarks.part(points[4]).x, landmarks.part(points[4]).y),
                        (landmarks.part(points[5]).x, landmarks.part(points[5]).y)], np.int32)

    # a mask of same size of gray frame, filling it with 0
    height, width, _ = image.shape
    mask = np.zeros((height,width), np.uint8)

    # draw a polygon on the mask according to the landmarks and fill it with white
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    # peforming pixel wise and, only gives us the eye-region from the original gray frame
    eye = cv2.bitwise_and(gray_image, gray_image, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # isolating the eye using landmarks from eye_region
    gray_eye = eye[min_y: max_y, min_x: max_x]

    # performing median blur to reduce noise, gaussian blur can also be used
    gray_eye = cv2.medianBlur(gray_eye, 5)

    # center of gray_eye
    eyeCenter = getMidPoint((0,0), (gray_eye.shape[1], gray_eye.shape[0]))

    # detecting HOUGH circles in the current frame
    circles = cv2.HoughCircles(gray_eye, cv2.HOUGH_GRADIENT, 1, 20,
              param1=30,
              param2=15,
              minRadius=0,
              maxRadius=0)

    try:
        # returning the coordinates of the circle and the center
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0,:]:
            
            return [(x, y), eyeCenter]
    
    except:
        print("No circles found")
    
    finally:
        print("Continuing")
    
    return []


# function to move the cursor based on distance between the center and the iris
def moveCursor(right, left):

    if right and left:
        rightDiff = tuple(map(operator.sub, right[1], right[0]))
        leftDiff = tuple(map(operator.sub, left[1], left[0]))
        sum = tuple(map(operator.add, rightDiff, leftDiff))
        meanDiff = (int(sum[0] / 2), int(sum[1] / 2))

    elif right:
        meanDiff = tuple(map(operator.sub, right[1], right[0]))

    elif left:
        meanDiff = tuple(map(operator.sub, left[1], left[0]))

    else:
        meanDiff = (0,0)

    cursor.moveRel(meanDiff[0] * 25, meanDiff[1] * 25, duration = 1)
    return


# getting the webcam ready 
fvs = FileVideoStream(0).start()
time.sleep(1.0)

while fvs.more():

    # reading frame and converting it to grayscale
    image = fvs.read()
    image = imutils.resize(image, width=450)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # face detection using HOG->SVM
    faces = face_detector(gray_image)

    for face in faces:
        # getting the landmarks
        landmarks = getLandmarks(gray_image, face)

        rightDiff = getDiff([42,43,44,45,46,47], landmarks, image, gray_image)
        leftDiff = getDiff([36,37,38,39,40,41], landmarks, image, gray_image)

        moveCursor(rightDiff, leftDiff)

    # displaying the frame
    cv2.imshow("frame", image)
    if cv2.waitKey(1) == ord('q'):
        break

# exit with all resources
webcam.release()
cv2.destroyAllWindows()