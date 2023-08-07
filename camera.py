import cvzone.HandTrackingModule
import cv2
import mediapipe as mp
import time
import mouse

cap = cv2.VideoCapture(0)
detector = cvzone.HandTrackingModule.HandDetector(maxHands=1, detectionCon=0.7)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands1, img = detector.findHands(img)
    if hands1:
        m = hands1[0]
        fingers = detector.fingersUp(m)
        sum = fingers.count(1)
        print(sum)
    results = hands.process(imgRGB)
    x =results.multi_hand_landmarks

    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)





    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break
    
    