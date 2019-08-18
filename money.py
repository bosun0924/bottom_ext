import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
def bars_region(image): 
    #get the resolution of the image
    height, width, channel = image.shape
    area = [(width*0.25, height),(width*0.25, height*0.9),(width*0.7, height*0.9),(width*0.7, height),]
    crop_area = np.array([area], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, (255,255,255))
    bottom_region = cv2.bitwise_and(image, mask)
    return bottom_region
'''
img = cv2.imread('test3.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#bottom_region = bars_region(img)
money_region = img[1045:1070, 1200:1280]
money_region_gray = cv2.cvtColor(money_region, cv2.COLOR_BGR2GRAY)
_, money_region_binary = cv2.threshold(money_region_gray, 200, 255, cv2.THRESH_BINARY)
#cv2.imwrite('./raw.png',cv2.resize(bottom_region_binary, (480, 180)))
#######got the money########
money = money_region_binary

#######get the circle on the right#######
#######right panel region: X (1120,1327), Y(930, 1080)

circles = cv2.HoughCircles(img_gray[930:1080, 1120:1327],cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=15,maxRadius=20)
if (circles is not None):
    print(circles)
else:
    print('no circles found')

########hsv detecting whether the skills have been used or not########
low_limit = np.array([0, 0, 128])
high_limit = np.array([180, 255, 255])
highlight = cv2.inRange(img_hsv, low_limit, high_limit)
##
skill_0 = highlight[950:990, 680:722]
skill_Q = highlight[950:1005, 731:787]
skill_W = highlight[950:1005, 798:853]
skill_E = highlight[950:1005, 861:921]
skill_R = highlight[950:1005, 930:990]
skill_D = highlight[950:990, 1005:1048]
skill_F = highlight[950:990, 1055:1100]
#CHECK THE AVERAGE BRIGHTNESS
print("_________mean brightness_________")
print(cv2.mean(skill_0))
print(cv2.mean(skill_Q))
print(cv2.mean(skill_W))
print(cv2.mean(skill_E))
print(cv2.mean(skill_R))
print(cv2.mean(skill_D))
print(cv2.mean(skill_F))
print("__________________________________")
#######
plt.figure()
plt.subplot(2,4,1)
plt.imshow(skill_0)
plt.subplot(2,4,2)
plt.imshow(skill_Q)
plt.subplot(2,4,3)
plt.imshow(skill_W)
plt.subplot(2,4,4)
plt.imshow(skill_E)
plt.subplot(2,4,5)
plt.imshow(skill_R)
plt.subplot(2,4,6)
plt.imshow(skill_D)
plt.subplot(2,4,7)
plt.imshow(skill_F)

plt.figure()
plt.imshow(img_gray)

plt.figure()
plt.imshow(img_hsv)

plt.figure()
plt.imshow(highlight)

plt.figure()
plt.imshow(img_gray[930:1080, 1120:1327])
plt.show()
