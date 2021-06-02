import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# hand detection model
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# drawing connections
mpDraw = mp.solutions.drawing_utils

previousTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)
    
    # calculating framerates
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime=currentTime

    cv2.putText(img, str(int(fps)),(10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    # print(success)
    cv2.imshow('image', img)
    
    # exit if pressed 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
exit(1)
