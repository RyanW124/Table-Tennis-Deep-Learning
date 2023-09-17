import cv2
greenLower = (0, 0, 150)
greenUpper = (360, 150, 255)
cam = cv2.VideoCapture("game_1.mp4")
cam.set(cv2.CAP_PROP_FPS, 10)

for i in range(50):
    ret, img = cam.read()

    img = cv2.resize(img, (400, 300))
    img = img[:, 0:200 ]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=85, threshold2=255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('i', mask)
    cv2.imshow('im', img)

    cv2.waitKey(0)

    cv2.destroyWindow('i')

    cv2.destroyWindow('im')

cv2.destroyAllWindows()
cam.release()
