import cv2
import numpy as np

def sobel(channel):
    sobelx = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, 3))
    sobely = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1, 3))
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    binary = np.ones_like(magnitude)
    binary[(magnitude >= 110) & (magnitude <= 255)] = 0
    return binary

def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = np.linalg.inv(M)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, inv_M