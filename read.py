import mediapipe as mp
import cv2 

## mediapipe declarations
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
## Open CV declaration
capture = cv2.VideoCapture('vsTagirR1.mp4')

## resize video frame defintion
def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:

    while True:
        isTrue, frame = capture.read()
        ## resize big frames
        frame_resized = rescaleFrame(frame)
        ## Recolor Feed for the holistic model
        image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        ## Make detections
        results = holistic.process(image)
        #print(results.face_landmarks)

    ##All landmarks

        ## Recolor Feed for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ## Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        ## Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


        cv2.imshow('resized', image)

        if cv2.waitKey(20) & 0xFF==ord('q'):
            break
capture.release()
cv2.destroyAllWindows()

