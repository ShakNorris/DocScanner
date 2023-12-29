import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract


vid_cap = cv2.VideoCapture(0 + cv2.CAP_ANY)

width, height = 800, 600

vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

count = 0

def doc_scanner(img):
    global document_contour

    document_contour = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_scale, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
                if area > max_area and len(approx) == 4: ## making sure contour has 4 corners
                    document_contour = approx
                    max_area = area

    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)


def doc_processing(img):
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_scale, 128, 255, cv2.THRESH_BINARY)

    return threshold


while True:
    success, frame = vid_cap.read()
    if frame is None:
        break
    frame_copy = frame.copy()

    doc_scanner(frame_copy)
    cv2.imshow('input', frame)

    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    pytesseract.image_to_string(warped)
    cv2.imshow('Document', warped)

    ## This part is only good for scanning text documents
    # processed = doc_processing(warped)
    # cv2.imshow('processed', processed)

    key_press = cv2.waitKey(33)
    if key_press == ord('q'):
        break
    if key_press == ord('s'):
        cv2.imwrite("output/scanned_" + str(count) + ".jpg", warped)
        count += 1

cv2.destroyAllWindows()

