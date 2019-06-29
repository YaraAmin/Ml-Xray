import numpy as np
import cv2

def background_removal(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(256,256))
    #apply threshold
    th, th3 = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
#   cv2.imshow("threshold",th3)
     #difine the kernal we will compare to input image matrices
    kernal = np.ones((5,5), np.uint8)
    #Erosion .. add pixels ro the boundries of object if the kernal "fits"
    erosion = cv2.erode(th3,kernal,iterations=2)
#   cv2.imshow('erosion',erosion)
    #Finding contours
    edges, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #Sort contours large to small
    sorted_contours= sorted(contours,key=cv2.contourArea,reverse=True)
    # cv2.drawContours(img, sorted_contours[0], -1, (255,255,255), 1, cv2.LINE_AA)
    # cv2.imshow("Result", img)
    #create the mask
    mask = np.zeros(img.shape[:2], dtype="uint8") * 255
    # Draw the contours on the mask and fill it with color
    mask = cv2.fillPoly(mask, pts =[sorted_contours[0]], color=255)
#   cv2.imshow("mask",mask)
    clear_img = cv2.bitwise_and(img,img, mask=mask)
#     print("image shape",img.shape)
    return clear_img
