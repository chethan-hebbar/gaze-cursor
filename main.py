import cv2
# dlib is a library that will help us perform shape-prediction(landmarks detection) on the image
import dlib 
import numpy as np
# pyautogui will help us control the cursor
import pyautogui as cursor

# midpoint of two points
def getMidPoint(p1, p2):
    return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)

# instantiating our face detector and dlib-68 shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# method to get the gaze-ratio and detect whether eye is looking left or right and up or down
def getGazeRatios(points, landmarks):

    #left_eye_region = np.array([(part.x,part.y) for part in landmarks.part()], np.int32)
    eye_region = np.array([(landmarks.part(points[0]).x, landmarks.part(points[0]).y),
                        (landmarks.part(points[1]).x, landmarks.part(points[1]).y),
                        (landmarks.part(points[2]).x, landmarks.part(points[2]).y),
                        (landmarks.part(points[3]).x, landmarks.part(points[3]).y),
                        (landmarks.part(points[4]).x, landmarks.part(points[4]).y),
                        (landmarks.part(points[5]).x, landmarks.part(points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height,width), np.uint8)

    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]

    threshold_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    bottom_threshold = threshold_eye[int(height / 2): height, 0: width]
    bottom_white = cv2.countNonZero(bottom_threshold)

    top_threshold = threshold_eye[0: int(height / 2), 0: width]
    top_white = cv2.countNonZero(top_threshold)

    vertical_gaze_ratio = bottom_white / top_white

    horizontal_gaze_ratio = left_side_white / right_side_white

    return (horizontal_gaze_ratio, vertical_gaze_ratio)


# utility functions to get gaze directions
def horizontal(horizontal_ratio):
    # -1 = left, 1 = right, 0 = neutral

    if horizontal_ratio < 0.1:
        return 1
    
    #elif horizontal_ratio > 1.1:
        #return -1

    else:
        return 0

def vertical(vertical_ratio):
    # -1 = up, 1 = down, 0 = neutral

    if vertical_ratio < 0.3:
        return 1
    
    #elif vertical_ratio > 2.0:
        #return -1

    else:
        return 0


# computes the final gaze direction
def moveCursor(horizontal, vertical):

    if(horizontal == 0 and vertical == 0):
        return

    elif(horizontal == 0 and vertical == 1):
        cursor.moveRel(0, 100, duration = 2)
        return

    elif(horizontal == 0 and vertical == -1):
        cursor.moveRel(0, -100, duration = 2)
        return

    elif(horizontal == 1 and vertical == 0):
        cursor.moveRel(100, 0, duration = 2)
        return

    elif(horizontal == -1 and vertical == 0):
        cursor.moveRel(-100, 0, duration = 2)
        return

    else:
        return

# setting up video capture
cap = cv2.VideoCapture(0)

while(True):

    # reading from the webcam
    _, frame = cap.read()

    # converting frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting faces in the gray-frame
    faces = detector(gray)

    for face in faces:

        # using shape predictor to detect the 68 point face in frame
        landmarks = predictor(gray, face)

        # gaze ratios for both the eyes
        left_ratios = getGazeRatios([36,37,38,39,40,41], landmarks)
        right_ratios = getGazeRatios([42,43,44,45,46,47], landmarks)

        # getting their averages
        horizontal_ratio = (left_ratios[0] + right_ratios[0]) / 2
        vertical_ratio = (left_ratios[1] + right_ratios[1]) / 2

        # controlling the cursor according to gaze-ratios, values were taken by observations
        horizontal_dir = horizontal(horizontal_ratio)
        vertical_dir = vertical(vertical_ratio)

        # moving the cursor accordingly
        moveCursor(horizontal_dir, vertical_dir)

    # ending the capture if 'q' is hit
    if cv2.waitKey(1) == ord('q'):
        break

# closing all windows upon exit()
cap.release()



