import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

happy_img = cv2.imread("images/png.0.happy.jpg")
sad_img = cv2.imread("images/png.1.sad.jpg")
angry_img = cv2.imread("images/png.2.angry.jpg")
surprised_img = cv2.imread("images/png.3.surprised.jpg")
love_img = cv2.imread("images/png.4.love.jpg")
peace_img = cv2.imread("images/png.5.peace.png")

def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    background[y:y+h, x:x+w] = overlay
    return background

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        totalFingers = fingers.count(1)

        emotion = "Neutral"
        overlay = None

        if totalFingers == 1:
            emotion = "Happy"
            overlay = happy_img
        elif totalFingers == 2:
            emotion = "Sad"
            overlay = sad_img
        elif totalFingers == 3:
            emotion = "Angry"
            overlay = angry_img
        elif totalFingers == 4:
            emotion = "Surprised"
            overlay = surprised_img
        elif totalFingers == 5:
            emotion = "Love"
            overlay = love_img
        elif totalFingers == 0:
            emotion = "Peace"
            overlay = peace_img

        if overlay is not None:
            overlay = cv2.resize(overlay, (150, 150))
            img = overlay_image(img, overlay, 10, 10)

        cv2.putText(img, emotion, (50, 450), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)

    cv2.imshow("Hand Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()