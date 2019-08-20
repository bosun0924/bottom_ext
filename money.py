import cv2
import matplotlib.pyplot as plt
import numpy as np
import TrainAndTest

cap = cv2.VideoCapture('./test.mp4')
shot = np.zeros((50, 160))
frame_cntr = 0#frame counter to exract 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))#erosion kernal
while(cap.isOpened()):
    ret, img = cap.read()
    if (ret == False):
        break
    else:
        frame_cntr = frame_cntr+1
        money_region = img[1045:1070, 1200:1280]
        money_region = cv2.resize(money_region, (160, 50))
        money_region_gray = cv2.cvtColor(money_region, cv2.COLOR_BGR2GRAY)
        _, money_region_binary = cv2.threshold(money_region_gray, 
                                                100, #threshold = 200
                                                255, #max value in image
                                                cv2.THRESH_BINARY)#binary thresh
        #######got the money########
        money = money_region_binary
        erosion = cv2.erode(money,kernel)
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('money',erosion)
        number = TrainAndTest
        if (frame_cntr == 30):
            shot = np.concatenate((shot, closing), axis=0)
            print(number)
            frame_cntr = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.imwrite('./money.png',shot)
cap.release()
cv2.destroyAllWindows()