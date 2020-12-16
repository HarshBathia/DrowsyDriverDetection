import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream # For pi
import dlib


def flip(img, axes):
    """Flipping function."""
    if (axes == 0):
        return cv2.flip(img, 0)
    elif(axes == 1):
        return cv2.flip(img, 1)
    elif(axes == -1):
        return cv2.flip(img, -1)


EYE_AR_THRESH = 0.3  # Max threshold for which eye is considered open
EYE_AR_CONSEC_FRAMES = 7  # Max frames for EAR to be lower than EYE_AR_THRESH
MOUTH_AR_THRESH = 0.4  # Max threshold for mouth
SHOW_POINTS_FACE = False
SHOW_CONVEX_HULL_FACE = True
SHOW_INFO = True
# Counters
ear = 0
mar = 0
COUNTER_FRAMES_EYE = 0
COUNTER_FRAMES_MOUTH = 0
COUNTER_BLINK = 0
COUNTER_MOUTH = 0


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


videoSteam = cv2.VideoCapture(0)
# videoSteam = VideoStream(usePiCamera=True).start() # For Pi
ret, frame = videoSteam.read()
size = frame.shape
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


t_end = time.time()
while True:
    ret, frame = videoSteam.read()
    frame = flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        jaw = shape[48:61]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(jaw)

        if SHOW_CONVEX_HULL_FACE:
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [leftEyeHull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [jawHull], 0, (255, 255, 255), 1)
        # if COUNTER_BLINK > 1500 or COUNTER_MOUTH > 2000:
        #     cv2.putText(frame, "Send Alert!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if ear < EYE_AR_THRESH:
            COUNTER_FRAMES_EYE += 1
            if COUNTER_FRAMES_EYE >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Sleeping Driver!", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if COUNTER_FRAMES_EYE > 2:
                COUNTER_BLINK += 1
            COUNTER_FRAMES_EYE = 0

        if mar >= MOUTH_AR_THRESH:
            COUNTER_FRAMES_MOUTH += 1
        else:
            if COUNTER_FRAMES_MOUTH > 5:
                COUNTER_MOUTH += 1

            COUNTER_FRAMES_MOUTH = 0

        if (time.time() - t_end) > 60:
            t_end = time.time()
            COUNTER_BLINK = 0
            COUNTER_MOUTH = 0

    if SHOW_INFO:
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (200, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Blinks: {}".format(COUNTER_BLINK), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Mouths: {}".format(COUNTER_MOUTH), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        SHOW_CONVEX_HULL_FACE = not SHOW_CONVEX_HULL_FACE
    if key == ord('i'):
        SHOW_INFO = not SHOW_INFO
    time.sleep(0.02)

videoSteam.release()
cv2.destroyAllWindows()
