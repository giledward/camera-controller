import cvzone.HandTrackingModule
import cv2
import mediapipe as mp
import time
import mouse

cap = cv2.VideoCapture(0)
detector = cvzone.HandTrackingModule.HandDetector(maxHands=1, detectionCon=0.7)
mode = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

def modeselector(sum):
    print("what mode you want to use?")
    time.sleep(1)
    mode = sum

modeselector(sum)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands1, img = detector.findHands(img)
    if hands1:
        m = hands1[0]
        fingers = detector.fingersUp(m)
        sum = fingers.count(1)
        
    else: 
        mode =0
    results = hands.process(imgRGB)
    x =results.multi_hand_landmarks
    
    
    if mode == 1:
        print("you are in mode 1")
    elif mode == 2:
        print("you are in mode 2")
    elif mode == 3:
        print("you are in mode 3")
    elif mode == 4:
        print("you are in mode 4")
    elif mode == 5:
        print("you are in mode 5")
    else:
        print("select a mode")
    
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)






    