# Python program to demonstrate erosion and  
# dilation of images. 
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Reading the input image 
img = cv2.imread('t2.png', 0)

canv = np.ones(img.shape)
# Taking a matrix of size 5 as the kernel 
kernel = np.ones((10, 10), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=4)

edged = cv2.Canny(img_erosion, 30, 200)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
max_perim = -1
max_perim_idx = -1

for i in range(len(contours)):
    if cv2.arcLength(contours[i], True) > max_perim:
        max_perim = cv2.arcLength(contours[i], True)
        max_perim_idx = i

arr = cv2.drawContours(canv, contours, max_perim_idx, (0, 255, 0), 1)
arr *= 255
image = Image.fromarray(arr).convert('RGB')
ImageDraw.floodfill(image, xy=(0,0),value=(255,0,255),thresh=200)
n  = np.array(image)
n[(n[:, :, 0:3] != [255,0,255]).any(2)] = [0,0,0]
n[(n[:, :, 0:3] == [255,0,255]).all(2)] = [255,255,255]

kernel2 = np.ones((5,5),np.float32)/25
n = cv2.filter2D(n,-1,kernel2)

cv2.imshow('Contours', np.asarray(n))
cv2.waitKey(0)
cv2.destroyAllWindows()