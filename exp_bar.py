import cv2
import matplotlib.pyplot as plt
import numpy as np

def bars_region(image): 
    #get the resolution of the image
    height, width, channel = image.shape
    area = [(width*0.25, height),(width*0.25, height*0.85),(width*0.7, height*0.85),(width*0.7, height),]
    crop_area = np.array([area], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, (255,255,255))
    bottom_region = cv2.bitwise_and(image, mask)
    return bottom_region

img = cv2.imread('./test_pictures/test2.png')
bars_area = bars_region(img)
#head_area = img[920:1080, 520:680]
bars_gray = cv2.cvtColor(bars_area, cv2.COLOR_BGR2GRAY)
bars_hsv = cv2.cvtColor(bars_area, cv2.COLOR_BGR2HSV)
#bars_gray_blur = cv2.GaussianBlur( bars_gray, (3, 3), 2, 2 )

#Threshold the gray image

_, bars_region = cv2.threshold(bars_gray, 30, 127, cv2.THRESH_BINARY)
#head_region = cv2.inRange(head_hsv, (180, 0,0), (300,100,100))

circles = cv2.HoughCircles(bars_gray,cv2.HOUGH_GRADIENT,1,2,minRadius=5,maxRadius=8)

print(type(circles))
if (circles is not None):
    circles = np.uint16(np.around(circles)) 
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(img)
plt.show()
'''
plt.figure()
plt.imshow(bars_area)

plt.figure()
plt.imshow(bars_region)

plt.show()
'''