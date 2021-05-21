import cv2
from forked import GazeTracking
import pyautogui as cursor

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    #new_frame = gaze.annotated_frame()

    if gaze.is_right():
        cursor.moveRel(50, 0, duration = 1)

    elif gaze.is_left():
        cursor.moveRel(-50, 0, duration = 1)

    else:
        verticalRatio = gaze.vertical_ratio()

        if(verticalRatio):
            if verticalRatio > 0.75:
                cursor.moveRel(0, 50, duration = 1)

            elif verticalRatio < 0.25:
                cursor.moveRel(0, -50, duration = 1)
    
            else:
                cursor.moveRel(0, 0, duration = 1)
        
        else:
            cursor.moveRel(0, 0, duration = 1)
            

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()