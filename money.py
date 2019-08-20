import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('./test.mp4')
shot = np.zeros((25,80))
frame_cntr = 0
while(cap.isOpened()):
    ret, img = cap.read()
    if (ret == False):
        break
    else:
        frame_cntr = frame_cntr+1
        money_region = img[1045:1070, 1200:1280]
        money_region_gray = cv2.cvtColor(money_region, cv2.COLOR_BGR2GRAY)
        _, money_region_binary = cv2.threshold(money_region_gray, 
                                                200, #threshold = 200
                                                255, #max value in image
                                                cv2.THRESH_BINARY)#binary thresh
        #######got the money########
        money = money_region_binary
        cv2.imshow('money',money_region_binary)
        if (frame_cntr == 25):
            shot = np.concatenate((shot, money_region_binary), axis=0)
            frame_cntr = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.imwrite('./money.png',shot)
cap.release()
cv2.destroyAllWindows()