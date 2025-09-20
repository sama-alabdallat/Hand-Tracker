import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        totalFingers = fingers.count(1)
        print(totalFingers)

        cv2.putText(img, f'{totalFingers}', (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

    cv2.imshow("Hand Tracker", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break