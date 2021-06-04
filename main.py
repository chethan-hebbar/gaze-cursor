import cv2 
import dlib 
import numpy as np
import sys
import pyautogui as cursor
import operator

# loading our face-detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
getLandmarks = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# function to get the midpoint between two points
def getMidPoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

# function to draw points on landmarks
def drawEyes(image, landmarks, points):
    # drawing points on the eye landmarks to verify if predictor is working well
    for point in points:
        cv2.circle(image, (landmarks.part(point).x, landmarks.part(point).y), 0, (0, 0, 255), thickness = -1)
    
    return image

# function returns the position of center of iris and position of center of the eye
def getDiff(points, landmarks, gray_image, size):

    # getting the eye with landmarks
    eye_region = np.array([(landmarks.part(points[0]).x,landmarks.part(points[0]).y),
                        (landmarks.part(points[1]).x, landmarks.part(points[1]).y),
                        (landmarks.part(points[2]).x, landmarks.part(points[2]).y),
                        (landmarks.part(points[3]).x, landmarks.part(points[3]).y),
                        (landmarks.part(points[4]).x, landmarks.part(points[4]).y),
                        (landmarks.part(points[5]).x, landmarks.part(points[5]).y)], np.int32)

    # a mask of same size of gray frame, filling it with 0
    mask = np.zeros(size, np.uint8)

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
        for (x,y,r) in circles[0,:]:
            
            return [(x,y),eyeCenter]
    
    except:
        print("No circles found")
    
    finally:
        print("Continuing")
    
    return []



# function to move the cursor based on distance between the center and the iris
def moveCursor(right, left):

    # if circles are detected on the both the left and right eyes, we get their average to move the cursor
    if right and left:
        rightDiff = tuple(map(operator.sub, right[1], right[0]))
        leftDiff = tuple(map(operator.sub, left[1], left[0]))
        sum = tuple(map(operator.add, rightDiff, leftDiff))
        meanDiff = (int(sum[0] / 2), int(sum[1] / 2))

    # if circle is present only on the right eye
    elif right:
        meanDiff = tuple(map(operator.sub, right[1], right[0]))

    # if circle is present only on the left eye
    elif left:
        meanDiff = tuple(map(operator.sub, left[1], left[0]))

    # else if no circle is detected we dont move the cursor
    else:
        meanDiff = (0,0)

    # move the cursor, scale magnified by 25
    cursor.moveRel(meanDiff[0] * 25, meanDiff[1] * 25, duration = 1)
    return


# getting the webcam ready 
webcam = cv2.VideoCapture(0)
# getting the params we need to monitor frame rate and to store the sizes
resize = 360
frameSkip = 3
frameCount = 0
time = cv2.getTickCount()
fps = 30 #random assignment

# getting the size values and scale values
if webcam.isOpened():
    ret,img = webcam.read()
    if ret == True:
        height = img.shape[0]
        frameScale = float(height)/resize
        size = img.shape[0:2]
    else:
        sys.exit()

# loop till cam closes
while webcam.isOpened():

    # reading frame and converting it to grayscale
    ret,image = webcam.read()

    # if the image is read, keep going
    if ret == True:

        # resetting timer when frameCount is reset
        if frameCount == 0:
            time = cv2.getTickCount()

        # rgb and gray images of the read frame
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # resizing and converting the image to rgb for our detector
        imageResize = cv2.resize(image, None, fx = 1.0/frameScale, fy = 1.0/frameScale)
        rgb_image = cv2.cvtColor(imageResize, cv2.COLOR_BGR2RGB)

        # if frame count is a multiple of frameSkip(frequency of frames)
        if(frameCount % frameSkip == 0):
            # face detection using HOG->SVM
            faces = face_detector(rgb_image, 0)
        
        # go through the faces
        for face in faces:
            # rescaling the (x,y,h,w) of the face according to our frameScale
            newFace = dlib.rectangle(int(face.left() * frameScale),
                               int(face.top() * frameScale),
                               int(face.right() * frameScale),
                               int(face.bottom() * frameScale))
            
            # getting the landmarks
            landmarks = getLandmarks(imageRGB, newFace)
            
            # to check the landmarks:: image = drawEyes(image, landmarks, [36,37,38,39,40,41,42,43,44,45,46,47])

            # getting the coords of the center of circle and center of eye(with landmarks) for both eyes
            rightEye = getDiff([42,43,44,45,46,47], landmarks, gray_image, size)
            leftEye = getDiff([36,37,38,39,40,41], landmarks, gray_image, size)

            # move the cursor
            moveCursor(rightEye, leftEye)

        # putting the fps on the image
        cv2.putText(image, "{0:.2f}-framePerSecond".format(fps), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
        cv2.imshow("frame", image)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # increment frame counter
        frameCount = frameCount + 1
        # calculate framePerSecond at an interval of 100 frames
        if (frameCount == 100):
            time = (cv2.getTickCount() - time)/cv2.getTickFrequency()
            fps = 100.0/time
            frameCount = 0

    else:
        sys.exit()

# exit with all resources
webcam.release()
cv2.destroyAllWindows()